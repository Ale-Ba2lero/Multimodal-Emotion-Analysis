import torch
import torch.nn as nn
import torch.nn.functional as F


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