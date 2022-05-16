import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=512, ch=64):
        super(ImageEncoder, self).__init__()
        self.ch = ch
        self.z_dim = z_dim
        self.features_output = self.ch * 8 * 2 * 2
        
        # input image size = 64 * 64
        self.features = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 32 * 32
            nn.Conv2d(ch, ch * 2, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(ch * 2), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 16 * 16
            nn.Conv2d(ch * 2, ch * 4, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(ch * 4), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 8 * 8
            nn.Conv2d(ch * 4, ch * 8, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 4 * 4
            nn.Conv2d(ch * 8, ch * 8, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 2 * 2
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
    
    
class ImageDecoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=512, ch=64):
        super(ImageDecoder, self).__init__()
        
        self.ch = ch
        
        self.upsample = nn.Sequential(
            nn.Linear(z_dim, ch * 8 * 2 * 2),
            nn.ReLU())
        
        self.hallucinate = nn.Sequential(
            nn.Conv2d(ch * 8, ch * 8, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(ch * 8, ch * 4, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(ch * 4), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(ch * 4, ch * 2, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(ch * 2), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(ch * 2, ch, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(ch, 3, 3, 1, 1),
            nn.Upsample(scale_factor = 2, mode = "nearest"))

    def forward(self, z):
        z = self.upsample(z)
        z = z.view(-1, self.ch * 8, 2, 2)
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
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=0)
        )
        
    def forward(self, z):
        return self.net(z)