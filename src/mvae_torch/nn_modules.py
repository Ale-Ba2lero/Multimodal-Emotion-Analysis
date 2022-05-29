import torch
import torch.nn as nn
import torch.nn.functional as F    


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * torch.sigmoid(x)

'''
class FaceEncoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=512, num_filters=128):
        super(FaceEncoder, self).__init__()
        self.nf = num_filters
        self.z_dim = z_dim
        self.features_output = self.nf * 8 * 32 * 32
        
        self.features = nn.Sequential(
            nn.Conv2d(3, self.nf, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf), nn.ReLU(),
            nn.Conv2d(self.nf, self.nf * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf*2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf*4), nn.ReLU(),
            nn.Conv2d(self.nf*4, self.nf * 8, 3, 1, 1, bias=True), nn.ReLU()
        )
        
        self.z_loc_layer = nn.Sequential(
            nn.Linear(self.features_output, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, z_dim))
        
        self.z_scale_layer = nn.Sequential(
            nn.Linear(self.features_output, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, z_dim))
        
    def forward(self, image):
        hidden = self.features(image)
        hidden = hidden.view(-1, self.features_output)
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale
    
    
class FaceDecoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=512, num_filters=128):
        super(FaceDecoder, self).__init__()
        self.nf = num_filters
        self.z_dim = z_dim
        
        self.linear =  nn.Sequential(
            nn.Linear(z_dim, self.nf * 8 * 32 * 32),
            nn.ReLU())
        
        self.hallucinate = nn.Sequential(
            nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf * 4), nn.ReLU(),
            nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf * 2), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf), nn.ReLU(),
            nn.Conv2d(self.nf, 3, 3, 1, 1, bias=True), nn.ReLU()
        ) 

    def forward(self, z):
        z = self.linear(z)
        z = z.view(-1, self.nf * 8, 32, 32)
        image = self.hallucinate(z) 
        return image'''
    
class FaceEncoder(nn.Module):
    """Parametrizes q(z|x).
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, n_latents):
        super(FaceEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class FaceDecoder(nn.Module):
    """Parametrizes p(x|z). 
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, n_latents):
        super(FaceDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 256 * 5 * 5),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False))

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5)
        z = self.hallucinate(z)
        return z  # NOTE: no sigmoid here. See train.py


class EmotionEncoder(nn.Module):
    def __init__(self, input_dim, z_dim=64, hidden_dim=512, use_cuda=True):
        super(EmotionEncoder, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Linear(input_dim, hidden_dim)
        
        self.z_loc_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim))
        
        self.z_scale_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim))
        self.z_dim = z_dim

    def forward(self, emotion):
        emotion = torch.nn.functional.one_hot(emotion, num_classes=self.input_dim)
        hidden = self.net(emotion.to(torch.float64))
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale


class EmotionDecoder(nn.Module):
    def __init__(self, output_dim, z_dim=64, hidden_dim=512, use_cuda=True):
        super(EmotionDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.net(z)