import torch
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist


import torch.nn as nn
import torch.nn.functional as F


# helper functions
class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * torch.sigmoid(x)

    
def swish(x):
    return x * torch.sigmoid(x)


class ProductOfExperts(nn.Module):
    def forward(self, loc, scale, eps=1e-8):
        scale = scale + eps # numerical constant for stability
        T = 1. / scale
        product_loc = torch.sum(loc * T, dim=0) / torch.sum(T, dim=0)
        product_scale = 1. / torch.sum(T, dim=0)
        return product_loc, product_scale
     
        
class ImageEncoder(nn.Module):
    def __init__(self, z_dim=64, ch=64, scale=2):
        super(ImageEncoder, self).__init__()
        self.ch = ch
        self.z_dim = z_dim
        self.features_output = self.ch * 8 * 2 * 2
        
        # input = 64 * 64
        self.features = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.MaxPool2d(scale, scale), # 32
            nn.Conv2d(ch, ch * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.ReLU(),
            nn.MaxPool2d(scale, scale), # 16
            nn.Conv2d(ch * 2, ch * 4, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(),
            nn.MaxPool2d(scale, scale), # 8
            nn.Conv2d(ch * 4, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.MaxPool2d(scale, scale), # 4
            nn.Conv2d(ch * 8, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.MaxPool2d(scale, scale), # 2
        )
                           
        self.z_loc_layer = nn.Sequential(
            nn.Linear(self.features_output, 256),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(256, z_dim))
        
        self.z_scale_layer = nn.Sequential(
            nn.Linear(self.features_output, 256),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(256, z_dim))
        

    def forward(self, image):
        hidden = self.features(image)
        hidden = hidden.view(-1, self.features_output)
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale
    
    
class ImageDecoder(nn.Module):
    def __init__(self, z_dim=64, ch=64, scale=2):
        super(ImageDecoder, self).__init__()
        
        self.ch = ch
        
        self.upsample = nn.Sequential(
            nn.Linear(z_dim, ch * 8 * 2 * 2),
            Swish())
        
        self.hallucinate = nn.Sequential(
            nn.Conv2d(ch * 8, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.Upsample(scale_factor = scale, mode = "nearest"),
            nn.Conv2d(ch * 8, ch * 4, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(),
            nn.Upsample(scale_factor = scale, mode = "nearest"),
            nn.Conv2d(ch * 4, ch * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.ReLU(),
            nn.Upsample(scale_factor = scale, mode = "nearest"),
            nn.Conv2d(ch * 2, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Upsample(scale_factor = scale, mode = "nearest"),
            nn.Conv2d(ch, 3, 3, 1, 1),
            nn.Upsample(scale_factor = scale, mode = "nearest"))

    def forward(self, z):
        z = self.upsample(z)
        z = z.view(-1, self.ch * 8, 2, 2)
        image = self.hallucinate(z)
        return image 


class EmotionEncoder(nn.Module):
    def __init__(self, z_dim, input_dim, use_cuda=True):
        super(EmotionEncoder, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Linear(input_dim, 256)
        
        self.z_loc_layer = nn.Sequential(
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, z_dim))
        
        self.z_scale_layer = nn.Sequential(
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, z_dim))
        self.z_dim = z_dim

    def forward(self, emotion):
        emotion = torch.stack([F.one_hot(emo, self.input_dim) for emo in emotion])
        hidden = self.net(emotion.to(torch.float64))
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale


class EmotionDecoder(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(EmotionDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            Swish(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=0)
        )
        
    def forward(self, z):
        return self.net(z)


class MVAE(nn.Module):
    def __init__(self, z_dim, emotion_dim, img_size=128, ch_size=64, use_cuda=True):
        super(MVAE, self).__init__()
        
        self.z_dim = z_dim
        self.img_size = img_size
        self.experts = ProductOfExperts()
        self.image_encoder = ImageEncoder(z_dim, ch=ch_size)
        self.image_decoder = ImageDecoder(z_dim, ch=ch_size)
        self.emotion_encoder = EmotionEncoder(z_dim, emotion_dim)
        self.emotion_decoder = EmotionDecoder(z_dim, emotion_dim)
        
        self.use_cuda = use_cuda
        self.LAMBDA_IMAGES = 1.0
        self.LAMBDA_RATINGS = 50.0
        
        if self.use_cuda:
            self.cuda()
            
    def model(self, images=None, emotions=None, annealing_beta=1.0):
        pyro.module("mvae", self)
        
        batch_size = 0
        if images is not None:
            batch_size = images.size(0)
        elif emotions is not None:
            batch_size = emotions.size(0)
        
        with pyro.plate("data"):      
            # sample the latent z from the (constant) prior, z ~ Normal(0,I)
            #z_loc  = torch.zeros(size=[batch_size, self.z_dim])
            #z_scale = torch.exp(torch.zeros(size=[batch_size, self.z_dim]))     
            
            z_loc = torch.zeros(torch.Size((1, batch_size, self.z_dim))) + 0.5
            z_scale = torch.ones(torch.Size((1, batch_size, self.z_dim))) * 0.1
            
            if self.use_cuda:
                z_loc, z_scale = z_loc.cuda(), z_scale.cuda()
            
            # sample from prior (value will be sampled by guide when computing the ELBO)
            with poutine.scale(scale=annealing_beta):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale))

            # decode the latent code z
            img_loc = self.image_decoder.forward(z)
            emo_probs = self.emotion_decoder.forward(z)
            
            # score against actual images
            if images is not None:
                with poutine.scale(scale=self.LAMBDA_IMAGES):
                    pyro.sample("obs_img", dist.Bernoulli(img_loc), obs=images)
                    
            # score against actual emotion
            if emotions is not None:
                with poutine.scale(scale=self.LAMBDA_RATINGS):
                    pyro.sample("obs_emotion", dist.Categorical(probs=emo_probs), obs=emotions)

            return img_loc, emo_probs
        
    def guide(self, images=None, emotions=None, annealing_beta=1.0):
        pyro.module("mvae", self)
        
        batch_size = 0
        if images is not None:
            batch_size = images.size(0)
        elif emotions is not None:
            batch_size = emotions.size(0)
            
        with pyro.plate("data"):
            # use the encoder to get the parameters used to define q(z|x)
            
            z_loc = torch.zeros(torch.Size((1, batch_size, self.z_dim))) + 0.5
            z_scale = torch.ones(torch.Size((1, batch_size, self.z_dim))) * 0.1
            
            if self.use_cuda:
                z_loc, z_scale = z_loc.cuda(), z_scale.cuda()
                
            if images is not None:
                image_z_loc, image_z_scale = self.image_encoder.forward(images)
                z_loc = torch.cat((z_loc, image_z_loc.unsqueeze(0)), dim=0)
                z_scale = torch.cat((z_scale, image_z_scale.unsqueeze(0)), dim=0)
            
            if emotions is not None:
                emotion_z_loc, emotion_z_scale = self.emotion_encoder.forward(emotions)
                z_loc = torch.cat((z_loc, emotion_z_loc.unsqueeze(0)), dim=0)
                z_scale = torch.cat((z_scale, emotion_z_scale.unsqueeze(0)), dim=0)
            
            z_loc, z_scale = self.experts(z_loc, z_scale)
            
            # sample the latent z
            with poutine.scale(scale=annealing_beta):
                pyro.sample("z", dist.Normal(z_loc, z_scale))
                
                
    def forward(self, image=None, emotion=None):
        z_loc, z_scale  = self.infer(image, emotion)
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).independent(1))
        # reconstruct inputs based on that gaussian
        image_recon = self.image_decoder(z)
        rating_recon = self.emotion_decoder(z)
        return image_recon, rating_recon, z_loc, z_scale
    
    
    def infer(self, images=None, emotions=None):
        batch_size = 0
        if images is not None:
            batch_size = images.size(0)
        elif emotions is not None:
            batch_size = emotions.size(0)
            
        z_loc = torch.zeros(torch.Size((1, batch_size, self.z_dim))) + 0.5
        z_scale = torch.ones(torch.Size((1, batch_size, self.z_dim))) * 0.1
        
        if self.use_cuda:
            z_loc, z_scale = z_loc.cuda(), z_scale.cuda()

        if images is not None:
            image_z_loc, image_z_scale = self.image_encoder.forward(images)
            z_loc = torch.cat((z_loc, image_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, image_z_scale.unsqueeze(0)), dim=0)

        if emotions is not None:
            emotion_z_loc, emotion_z_scale = self.emotion_encoder.forward(emotions)
            z_loc = torch.cat((z_loc, emotion_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, emotion_z_scale.unsqueeze(0)), dim=0)

        z_loc, z_scale = self.experts(z_loc, z_scale)
        return z_loc, z_scale      

    def reconstruct_img(self, images):
        # encode image x
        z_loc, z_scale = self.image_encoder(images)
        z = dist.Normal(z_loc, z_scale).sample()
        img_loc = self.image_decoder.forward(z)
        return img_loc
 
    def reconstruct_img_nosample(self, images):
        # encode image x
        z_loc, z_scale = self.image_encoder(images)
        img_loc = self.image_decoder.forward(z_loc)
        return img_loc
    
    def emotion_classifier(self, images):
        z_loc, z_scale = self.image_encoder(images)
        z = dist.Normal(z_loc, z_scale).sample()
        return self.emotion_decoder.forward(z)
 
    def emoToZ(self, emotion):
        z_loc, z_scale = self.emotion_encoder.forward(emotion)
        return dist.Normal(z_loc, z_scale).sample()

    def zToEmo(self, z):
        return self.emotion_decoder.forward(z)