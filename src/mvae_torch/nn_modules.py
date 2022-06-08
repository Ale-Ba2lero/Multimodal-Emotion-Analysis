import torch
import torch.nn as nn
import torch.nn.functional as F    


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * torch.sigmoid(x)

#------------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=128):
        super(ImageEncoder, self).__init__()
        self.fc1 = nn.Linear(64*64*3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_means = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = self.fc1(x.view(-1, 64*64*3))
        h = self.fc2(h)
        return self.fc_means(h), self.fc_logvar(h)

class ImageDecoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=128):
        super(ImageDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 64*64*3)

    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        out = self.fc_out(h)
        out = out.view(-1, 3, 64, 64)
        return out
#------------------------------------------------------------------------------   
#------------------------------------------------------------------------------   
class CnnFaceEncoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=128, num_filters=128):
        super(FaceEncoder, self).__init__()
        self.nf = num_filters
        self.z_dim = z_dim
        self.features_output = self.nf * 4 * 8 * 8
        
        self.features = nn.Sequential(
            nn.Conv2d(3, self.nf, 3, 1, 1, bias=False), 
            nn.Dropout(p=0.1), nn.BatchNorm2d(self.nf), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.nf, self.nf * 2, 3, 1, 1, bias=False), 
            nn.Dropout(p=0.1), nn.BatchNorm2d(self.nf*2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 1, bias=False), 
            nn.Dropout(p=0.1), nn.BatchNorm2d(self.nf*4), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.nf*4, self.nf * 4, 3, 1, 1, bias=True), nn.ReLU()
        )
        
        self.z_loc_layer = nn.Sequential(
            nn.Linear(self.features_output, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, z_dim))
        
        self.z_scale_layer = nn.Sequential(
            nn.Linear(self.features_output, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, z_dim))
        
    def forward(self, image):
        hidden = self.features(image)
        hidden = hidden.view(-1, self.features_output)
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale
    
    
class CnnFaceDecoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=512, num_filters=128):
        super(FaceDecoder, self).__init__()
        self.nf = num_filters
        self.z_dim = z_dim
        
        self.linear =  nn.Sequential(
            nn.Linear(z_dim, self.nf * 4 * 8 * 8), nn.Dropout(p=0.3),
            nn.ReLU())
        
        self.hallucinate = nn.Sequential(
            nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False), 
            nn.Dropout(p=0.1), nn.BatchNorm2d(self.nf * 4), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False), 
            nn.Dropout(p=0.1), nn.BatchNorm2d(self.nf * 2), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False), 
            nn.Dropout(p=0.1), nn.BatchNorm2d(self.nf), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(self.nf, 3, 3, 1, 1, bias=True), nn.ReLU()
        ) 

    def forward(self, z):
        z = self.linear(z)
        z = z.view(-1, self.nf * 4, 8, 8)
        image = self.hallucinate(z) 
        return image

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class DCGAN_01FaceEncoder(nn.Module):
    """Parametrizes q(z|x).
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, z_dim, num_filters=16):
        super(DCGAN_01FaceEncoder, self).__init__()
        self.num_filters = num_filters
        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(num_filters, num_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters*2),
            Swish(),
            nn.Conv2d(num_filters*2, num_filters*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters*4),
            Swish(),
            nn.Conv2d(num_filters*4, num_filters*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters*8),
            Swish())
         # Here, we define two layers, one to give z_loc and one to give z_scale
        self.z_loc_layer = nn.Sequential(
            nn.Linear(num_filters * 8 * 5 * 5, 512), # it's 256 * 5 * 5 if input is 64x64.
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, z_dim))
        self.z_scale_layer = nn.Sequential(
            nn.Linear(num_filters * 8 * 5 * 5, 512), # it's 256 * 5 * 5 if input is 64x64.
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, z_dim))

    def forward(self, image):
        hidden = self.features(image)
        hidden = hidden.view(-1, self.num_filters * 8 * 5 * 5) # it's 256 * 5 * 5 if input is 64x64.
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden)) #add exp so it's always positive
        return z_loc, z_scale


class DCGAN_01FaceDecoder(nn.Module):
    """Parametrizes p(x|z). 
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, z_dim, num_filters=16):
        super(DCGAN_01FaceDecoder, self).__init__()
        self.num_filters = num_filters
        self.upsample = nn.Sequential(
            nn.Linear(z_dim, num_filters * 8 * 5 * 5),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            Swish(),
            nn.ConvTranspose2d(num_filters * 4,  num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            Swish(),
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            Swish(),
            nn.ConvTranspose2d(num_filters, 3, 4, 2, 1, bias=True))

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, self.num_filters * 8, 5, 5)
        z = self.hallucinate(z)
        return z  # NOTE: no sigmoid here. See train.py

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class DCGAN_02FaceEncoder(nn.Module):
    """Parametrizes q(z|x).
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, z_dim, num_filters=16):
        super(DCGAN_02FaceEncoder, self).__init__()
        self.num_filters = num_filters
        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, 4, 2, 1,bias=False),
            Swish(),
            nn.Conv2d(num_filters, num_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters*2),
            Swish(),
            nn.Conv2d(num_filters*2, num_filters*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters*4),
            Swish())
        
        self.z_loc_layer = nn.Sequential(
            nn.Linear(num_filters * 4 * 8 * 8, 512),
            Swish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, z_dim))
        self.z_scale_layer = nn.Sequential(
            nn.Linear(num_filters * 4 * 8 * 8, 512),
            Swish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, z_dim))

    def forward(self, image):
        hidden = self.features(image)
        hidden = hidden.view(-1, self.num_filters * 4 * 8 * 8)
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale


class DCGAN_02FaceDecoder(nn.Module):
    """Parametrizes p(x|z). 
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, z_dim, num_filters=16):
        super(DCGAN_02FaceDecoder, self).__init__()
        self.num_filters = num_filters
        self.upsample = nn.Sequential(
            nn.Linear(z_dim, num_filters * 4 * 8 * 8),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 4,  num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            Swish(),
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            Swish(),
            nn.ConvTranspose2d(num_filters, 3, 4, 2, 1, bias=True))

    def forward(self, z):
        z = self.upsample(z)
        z = z.view(-1, self.num_filters * 4, 8, 8)
        z = self.hallucinate(z)
        return z

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

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