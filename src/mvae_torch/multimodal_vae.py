import logging
from typing import Tuple

import torch
import torch.nn.functional as torch_functional
from torch.autograd import Variable

from torch_mvae_util import Expert

class MultimodalVariationalAutoencoder(torch.nn.Module):

    def __init__(
            self,
            au_encoder: torch.nn.Module,
            au_decoder: torch.nn.Module,
            face_encoder: torch.nn.Module,
            face_decoder: torch.nn.Module,
            emotion_encoder: torch.nn.Module,
            emotion_decoder: torch.nn.Module,
            feature_fusion_net: torch.nn.Module,
            modes: dict,
            latent_space_dim: int,
            expert: Expert = None,
            use_cuda: bool = True
    ) -> None:
        super(MultimodalVariationalAutoencoder, self).__init__()
        self._logger = logging.getLogger(MultimodalVariationalAutoencoder.__name__)
        
        self._modes: dict = modes
            
        if self._modes['au'] is not None:
            self._au_encoder: torch.nn.Module = au_encoder
            self._au_decoder: torch.nn.Module = au_decoder
        if self._modes['face'] is not None:
            self._face_encoder: torch.nn.Module = face_encoder
            self._face_decoder: torch.nn.Module = face_decoder
        if self._modes['emotion'] is not None:
            self._emotion_encoder: torch.nn.Module = emotion_encoder
            self._emotion_decoder: torch.nn.Module = emotion_decoder
        self._feature_fusion_net: torch.nn.Module = feature_fusion_net
            
        self._expert: Expert = expert
        self._latent_space_dim: int = latent_space_dim
        
        if expert is not None:
            self._use_expert = True
        else:
            self._use_expert = False

        # Train on GPU
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
    
    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd

    def loss_function(self, 
                      au: torch.Tensor,
                      faces: torch.Tensor,
                      emotions: torch.Tensor,
                      au_reconstruction: torch.Tensor,
                      faces_reconstruction: torch.Tensor,
                      emotions_reconstruction: torch.Tensor,
                      z_loc: torch.Tensor,
                      z_scale: torch.Tensor,
                      beta: float,
                      latent_sample: torch.Tensor
                     ) -> dict:
        """
        Computes the VAE loss function:
            KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        # First calculate the reconstruction loss
        reconstruction_loss = 0
        
        if self._modes["au"] is not None:
            if au is not None:
                au_reconstruction_loss: torch.Tensor = torch_functional.mse_loss(au_reconstruction, au)
            else: 
                au_reconstruction_loss: torch.Tensor = torch.Tensor([0.0])
            if self.use_cuda: faces_reconstruction_loss = faces_reconstruction_loss.cuda()
            au_reconstruction_loss = self._modes["au"].weight * au_reconstruction_loss
            reconstruction_loss += au_reconstruction_loss
            
        if self._modes["face"] is not None: 
            if faces is not None:
                faces_reconstruction_loss: torch.Tensor = torch_functional.mse_loss(faces_reconstruction, faces)
            else: 
                faces_reconstruction_loss: torch.Tensor = torch.Tensor([0.0])
            if self.use_cuda: faces_reconstruction_loss = faces_reconstruction_loss.cuda()
            faces_reconstruction_loss = self._modes["face"].weight * faces_reconstruction_loss
            reconstruction_loss += faces_reconstruction_loss
            
        if self._modes["emotion"] is not None:
            if emotions is not None:
                emotions_reconstruction_loss: torch.Tensor = torch_functional.cross_entropy(emotions_reconstruction, emotions)
            else:
                emotions_reconstruction_loss: torch.Tensor = torch.Tensor([0.0])
            if self.use_cuda: faces_reconstruction_loss = faces_reconstruction_loss.cuda()
            emotions_reconstruction_loss = self._modes["emotion"].weight * emotions_reconstruction_loss
            reconstruction_loss += emotions_reconstruction_loss

        # Calculate the KLD loss
        log_var = torch.log(torch.square(z_scale))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - z_loc ** 2 - log_var.exp(), dim=1), dim=0)
        #total_loss = reconstruction_loss + beta * kld_loss
        
        true_samples = Variable(
                torch.randn_like(latent_sample),
                requires_grad = False
            )
        
        # Calculate the KLD loss
        mmd_loss = self.compute_mmd(true_samples, latent_sample)
        
        # Calculate the Total Loss
        total_loss = reconstruction_loss + mmd_loss + beta * kld_loss
        
        return {
            "total_loss": total_loss, "reconstruction_loss": reconstruction_loss,
            "kld_loss": kld_loss, "mmd_loss":mmd_loss,
            "au_reconstruction_loss": au_reconstruction_loss,
            "faces_reconstruction_loss": faces_reconstruction_loss, 
            "emotions_reconstruction_loss": emotions_reconstruction_loss
        }
        

    def _extract_batch_size_from_data(self, 
                                      au:torch.Tensor = None, 
                                      faces:torch.Tensor = None, 
                                      emotions:torch.Tensor = None
                                     ) -> int:
        if au is not None:
            batch_size = au.shape[0]
        if faces is not None:
            batch_size = faces.shape[0]
        elif emotions is not None:
            batch_size = emotions.shape[0]
        else:
            batch_size = 0
        return batch_size
    

    def infer_latent(self, 
                     au: torch.Tensor, 
                     faces: torch.Tensor, 
                     emotions: torch.Tensor
                    ) -> Tuple[torch.Tensor, ...]:
        # Use the encoders to get the parameters used to define q(z|x).
        
        batch_size: int = self._extract_batch_size_from_data(
            au=au,
            faces=faces,
            emotions=emotions
        )
            
        if self._use_expert:
            return self.apply_expert(au, faces, emotions, batch_size)
        else:
            return self.features_fusion(au, faces, emotions, batch_size)
        
    
    def features_fusion(self, 
                        au: torch.tensor,
                        faces: torch.Tensor, 
                        emotions: torch.Tensor, 
                        batch_size: int
                       ) -> Tuple[torch.Tensor, ...]:
        # hardwired feature size
        # moreover I am using the same size for both features, this may not be optimal
        
        if faces is not None:
            face_features = self._face_encoder.forward(faces)
        else:
            face_features = torch.zeros(batch_size, self._face_encoder.features_size)
            
        if emotions is not None:
            emotion_features = self._emotion_encoder.forward(emotions)
        else:
            emotion_features = torch.zeros(batch_size, self._emotion_encoder.features_size)
            
        if self.use_cuda:
            face_features = face_features.cuda()
            emotion_features = emotion_features.cuda()
            
        return self._feature_fusion_net(face_features, emotion_features)
    
    
    def apply_expert(self, 
                     au: torch.tensor,
                     faces: torch.Tensor, 
                     emotions: torch.Tensor, 
                     batch_size: int
                    ) -> Tuple[torch.Tensor, ...]:
        # Initialize the prior expert.
        # We initialize an additional dimension, along which we concatenate all the different experts.
        #   self.experts() then combines the information from these different modalities by multiplying
        #   the Gaussians together.
        z_loc, z_scale = (
            torch.zeros([1, batch_size, self._latent_space_dim]),
            torch.ones([1, batch_size, self._latent_space_dim])
        )
        
        if self.use_cuda:
            z_loc = z_loc.cuda()
            z_scale = z_scale.cuda()
            
        if au is not None:
            au_z_loc, au_z_scale = self._au_encoder.forward(au)
            z_loc = torch.cat((z_loc, au_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, au_z_scale.unsqueeze(0)), dim=0)
        
        if faces is not None:
            face_z_loc, face_z_scale = self._face_encoder.forward(faces)
            z_loc = torch.cat((z_loc, face_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, face_z_scale.unsqueeze(0)), dim=0)

        if emotions is not None:
            emotion_z_loc, emotion_z_scale = self._emotion_encoder.forward(emotions)
            z_loc = torch.cat((z_loc, emotion_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, emotion_z_scale.unsqueeze(0)), dim=0)

        # Give the inferred parameters to the expert to arrive at a unique decision
        z_loc_expert, z_scale_expert = self._expert(z_loc, z_scale)

        return z_loc_expert, z_scale_expert
    

    def generate(self, latent_sample: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        face_reconstruction = self._face_decoder.forward(latent_sample)
        emotion_reconstruction = self._emotion_decoder.forward(latent_sample)

        return face_reconstruction, emotion_reconstruction

    def forward(self, 
                au: torch.tensor= None, 
                faces: torch.tensor= None, 
                emotions: torch.tensor= None
               ) -> Tuple[torch.Tensor, ...]:
        # Infer the latent distribution parameters
        z_loc, z_scale = self.infer_latent(
            au=au,
            faces=faces,
            emotions=emotions
        )
                
        # Sample from the latent space         
        epsilon: torch.Tensor = torch.randn_like(z_loc)
        latent_sample: torch.Tensor = z_loc + epsilon * z_scale
            
        # Reconstruct inputs based on that Gaussian sample
        au_reconstruction, face_reconstruction, emotions_reconstruction = self.generate(
            latent_sample=latent_sample
        )

        return au_reconstruction, face_reconstruction, emotions_reconstruction, z_loc, z_scale, latent_sample
    