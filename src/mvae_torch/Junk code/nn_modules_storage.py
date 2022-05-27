class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResDown, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale) # nn.MaxPool2d(scale, scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResUp, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor = scale, mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x
    

class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n
    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """
    def __init__(self, channels=3, ch=64, z_dim=512):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.conv1 = ResDown(channels, ch)  # 64
        self.conv2 = ResDown(ch, 2*ch)  # 32
        self.conv3 = ResDown(2*ch, 4*ch)  # 16
        self.conv4 = ResDown(4*ch, 8*ch)  # 8
        self.conv5 = ResDown(8*ch, 8*ch)  # 4
        self.conv_mu = nn.Conv2d(8*ch, z_dim, 2, 2)  # 2
        self.conv_log_var = nn.Conv2d(8*ch, z_dim, 2, 2)  # 2
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        
        mu = mu.view(-1, self.z_dim)
        log_var = log_var.view(-1, self.z_dim)
        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels=3, ch=64, z_dim=512):
        super(Decoder, self).__init__()
        self.z_dim=z_dim
        self.conv1 = ResUp(z_dim, ch*8)
        self.conv2 = ResUp(ch*8, ch*8)
        self.conv3 = ResUp(ch*8, ch*4)
        self.conv4 = ResUp(ch*4, ch*2)
        self.conv5 = ResUp(ch*2, ch)
        self.conv6 = ResUp(ch, ch//2)
        self.conv7 = nn.Conv2d(ch//2, channels, 3, 1, 1)

    def forward(self, x):   
        x = x.view(-1, self.z_dim, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x 
    
    

class ImageEncoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=512, ch=128):
        super(ImageEncoder, self).__init__()
        self.ch = filter_size
        self.z_dim = z_dim
        self.features_output = self.ch * 8 * 2 * 2
        
        # input image size = 64 * 64
        self.features = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),nn.BatchNorm2d(ch), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 32 * 32
            
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 16 * 16
            
            nn.Conv2d(ch*2, ch * 2, 3, 1, 1, bias=False),nn.BatchNorm2d(ch * 2), nn.ReLU(),
            nn.Conv2d(ch * 2, ch * 4, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 8 * 8
            
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(),
            nn.Conv2d(ch * 4, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 4 * 4
            
            nn.Conv2d(ch*8, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.Conv2d(ch * 8, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.MaxPool2d(2, 2), # img size = 2 * 2
            
            nn.Conv2d(ch * 8, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
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
    def __init__(self, z_dim=64, hidden_dim=512, ch=128):
        super(ImageDecoder, self).__init__()
        
        self.ch = ch
        
        self.upsample = nn.Sequential(
            nn.Linear(z_dim, ch * 8 * 2 * 2),
            nn.ReLU())
        
        self.hallucinate = nn.Sequential(
            nn.Conv2d(ch * 8, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.Conv2d(ch * 8, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"), # img size = 4 * 4
            
            nn.Conv2d(ch*8, ch * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(),
            nn.Conv2d(ch * 8, ch * 4, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"), # img size = 8 * 8
            
            nn.Conv2d(ch*4, ch * 4, 3, 1, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(),
            nn.Conv2d(ch * 4, ch * 2, 3, 1, 1, bias=False),nn.BatchNorm2d(ch * 2), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"), # img size = 16 * 16
            
            nn.Conv2d(ch*2, ch * 2, 3, 1, 1, bias=False),nn.BatchNorm2d(ch * 2), nn.ReLU(),
            nn.Conv2d(ch * 2, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"), # img size = 32 * 32
            
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, 3, 3, 1, 1, bias=False), nn.BatchNorm2d(3), nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = "nearest"), # img size = 64 * 64
            
            nn.Conv2d(3, 3, 3, 1, 1, bias=True),
        ) 

    def forward(self, z):
        z = self.upsample(z)
        z = z.view(-1, self.ch * 8, 2, 2)
        image = self.hallucinate(z)
        return image 