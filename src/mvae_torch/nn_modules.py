import torch
import torch.nn as nn
import torch.nn.functional as F    
    
class FaceEncoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=512, num_filters=128):
        super(FaceEncoder, self).__init__()
        self.nf = num_filters
        self.z_dim = z_dim
        self.features_output = self.nf*4 * 16 * 16
        
        self.features = nn.Sequential(
            nn.Conv2d(3, self.nf, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.nf, self.nf * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf*2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.nf*2, self.nf * 4, 3, 1, 1, bias=True), nn.ReLU()
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
            nn.Linear(z_dim, self.nf * 4 * 16 * 16),
            nn.ReLU())
        
        self.hallucinate = nn.Sequential(
            nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf * 2), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False), nn.BatchNorm2d(self.nf), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(self.nf, 3, 3, 1, 1, bias=True), nn.ReLU()
        ) 

    def forward(self, z):
        z = self.linear(z)
        z = z.view(-1, self.nf * 4, 16, 16)
        image = self.hallucinate(z) 
        return image


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