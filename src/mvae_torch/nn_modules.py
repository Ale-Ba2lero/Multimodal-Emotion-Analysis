import torch
import torch.nn as nn
import torch.nn.functional as F    


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * torch.sigmoid(x)

#------------------------------------------------------------------------------
class AUFeatureExtraction(nn.Module):
    def __init__(self, input_dim, features_size, hidden_dim=256):
        super(AUFeatureExtraction, self).__init__()
        self.input_dim = input_dim
        self.features_size = features_size
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, features_size),
        )

    def forward(self, au):
        features = self.net(au.to(torch.float64))
        return features
    
class FaceFeatureExtraction(nn.Module):
    def __init__(self, features_size=64, num_filters=64):
        super(FaceFeatureExtraction, self).__init__()
        self.num_filters = num_filters
        self.features_size = features_size
        
        self.cnn = nn.Sequential(
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
        
        self.features = nn.Sequential(
            nn.Linear(num_filters * 8 * 5 * 5, 512), # it's 256 * 5 * 5 if input is 64x64.
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, features_size))

    def forward(self, image):
        hidden = self.cnn(image)
        hidden = hidden.view(-1, self.num_filters * 8 * 5 * 5) # it's 256 * 5 * 5 if input is 64x64.
        features = self.features(hidden)
        return features
    
class EmotionFeatureExtraction(nn.Module):
    def __init__(self, input_dim, features_size=64):
        super(EmotionFeatureExtraction, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Linear(input_dim, 128)
        self.features_size=features_size
        
        self.features = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),nn.Dropout(p=0.2),
            nn.Linear(128, features_size))
        
    def forward(self, emotion):
        emotion = torch.nn.functional.one_hot(emotion, num_classes=self.input_dim)
        hidden = self.net(emotion.to(torch.float64))
        features = self.features(hidden)
        return features

class FeaturesFusion(nn.Module):
    def __init__(self, feature_size=64, z_dim=64, hidden_dim=128):
        super(FeaturesFusion, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_means = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, *features):
        features = torch.cat(features, 1)
        hidden = self.relu(self.fc1(features))
        hidden = self.relu(self.fc2(hidden))
        z_loc = self.fc_means(hidden)
        z_scale = self.fc_logvar(hidden)
        return z_loc, z_scale
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
            nn.BatchNorm2d(self.nf), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.nf, self.nf * 2, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(self.nf*2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(self.nf*4), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.nf*4, self.nf * 4, 3, 1, 1, bias=True), nn.ReLU()
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
    
    
class CnnFaceDecoder(nn.Module):
    def __init__(self, z_dim=64, num_filters=128):
        super(CnnFaceDecoder, self).__init__()
        self.nf = num_filters
        self.z_dim = z_dim
        
        self.linear =  nn.Sequential(
            nn.Linear(z_dim, self.nf * 4 * 8 * 8), nn.Dropout(p=0.3),
            nn.ReLU())
        
        self.hallucinate = nn.Sequential(
            nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(self.nf * 4), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(self.nf * 2), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(self.nf), nn.ReLU(),
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

class DCGANFaceEncoder(nn.Module):
    """Parametrizes q(z|x).
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, z_dim, num_filters=16):
        super(DCGANFaceEncoder, self).__init__()
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


class DCGANFaceDecoder(nn.Module):
    """Parametrizes p(x|z). 
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, z_dim, num_filters=16):
        super(DCGANFaceDecoder, self).__init__()
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
        return z  

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class AUEncoder(nn.Module):
    def __init__(self, input_dim, z_dim=64, hidden_dim=256):
        super(AUEncoder, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Linear(input_dim, hidden_dim)
        
        self.z_loc_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, z_dim))
        
        self.z_scale_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, z_dim))
        self.z_dim = z_dim
        

    def forward(self, au):
        hidden = self.net(au.to(torch.float64))
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale


class AUDecoder(nn.Module):
    def __init__(self, output_dim, z_dim=64, hidden_dim=256):
        super(AUDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU())
        
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid())
        
        
    def forward(self, z):
        hidden = self.net(z)
        out = self.hidden(hidden)
        return out
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class EmotionEncoder(nn.Module):
    def __init__(self, input_dim, z_dim=64, hidden_dim=512):
        super(EmotionEncoder, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Linear(input_dim, hidden_dim)
        
        self.z_loc_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim))
        
        self.z_scale_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim))
        self.z_dim = z_dim

    def forward(self, emotion):
        emotion = torch.nn.functional.one_hot(emotion, num_classes=self.input_dim)
        emotion = emotion.reshape(-1, self.input_dim).to(torch.float64)
        hidden = self.net(emotion)
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale


class EmotionDecoder(nn.Module):
    def __init__(self, output_dim, z_dim=64, hidden_dim=512):
        super(EmotionDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim, bias=False), 
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.net(z)