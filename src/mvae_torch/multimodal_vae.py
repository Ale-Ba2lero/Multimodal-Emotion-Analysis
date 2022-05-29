import logging
from typing import Tuple

import torch
import torch.nn.functional as torch_functional

from torch_mvae_util import Expert

class MultimodalVariationalAutoencoder(torch.nn.Module):

    def __init__(
            self,
            face_encoder: torch.nn.Module,
            face_decoder: torch.nn.Module,
            emotion_encoder: torch.nn.Module,
            emotion_decoder: torch.nn.Module,
            loss_weights: dict,
            expert: Expert,
            latent_space_dim: int,
            use_cuda: bool = False
    ) -> None:
        super(MultimodalVariationalAutoencoder, self).__init__()
        self._logger = logging.getLogger(MultimodalVariationalAutoencoder.__name__)

        self._face_encoder: torch.nn.Module = face_encoder
        self._face_decoder: torch.nn.Module = face_decoder
        self._emotion_encoder: torch.nn.Module = emotion_encoder
        self._emotion_decoder: torch.nn.Module = emotion_decoder

        self._loss_weights: dict = loss_weights
        self._expert: Expert = expert
        self._latent_space_dim: int = latent_space_dim

        # Train on GPU
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def loss_function(
            self,
            faces: torch.Tensor,
            emotions: torch.Tensor,
            faces_reconstruction: torch.Tensor,
            emotions_reconstruction: torch.Tensor,
            z_loc: torch.Tensor,
            z_scale: torch.Tensor,
            beta: float,
            ) -> dict:
        """
        Computes the VAE loss function:
            KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        # First calculate the reconstruction loss
        if faces is not None:
            faces_reconstruction_loss: torch.Tensor = torch_functional.mse_loss(faces_reconstruction, faces)
        else: 
            faces_reconstruction_loss: torch.Tensor = torch.Tensor([0.0])
                
        if emotions is not None:
            emotions_reconstruction_loss: torch.Tensor = torch_functional.cross_entropy(emotions_reconstruction, emotions)
        else:
            emotions_reconstruction_loss: torch.Tensor = torch.Tensor([0.0])

        if self.use_cuda:
            faces_reconstruction_loss = faces_reconstruction_loss.cuda()
            emotions_reconstruction_loss = emotions_reconstruction_loss.cuda()

        faces_reconstruction_loss = self._loss_weights["face"] * faces_reconstruction_loss
        emotions_reconstruction_loss = self._loss_weights["emotion"] * emotions_reconstruction_loss

        reconstruction_loss = faces_reconstruction_loss + emotions_reconstruction_loss

        # Calculate the KLD loss
        log_var = torch.log(torch.square(z_scale))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - z_loc ** 2 - log_var.exp(), dim=1), dim=0)
        total_loss = reconstruction_loss + beta * kld_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kld_loss": kld_loss,
            "faces_reconstruction_loss": faces_reconstruction_loss,
            "emotions_reconstruction_loss": emotions_reconstruction_loss
        }

    def _extract_batch_size_from_data(self, faces:torch.Tensor = None, emotions:torch.Tensor = None) -> int:
        if faces is not None:
            batch_size = faces.shape[0]
        elif emotions is not None:
            batch_size = emotions.shape[0]
        else:
            batch_size = 0

        return batch_size

    def infer_latent(self, faces: torch.Tensor, emotions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Use the encoders to get the parameters used to define q(z|x).
        # Initialize the prior expert.
        # We initialize an additional dimension, along which we concatenate all the different experts.
        #   self.experts() then combines the information from these different modalities by multiplying
        #   the Gaussians together.
        batch_size: int = self._extract_batch_size_from_data(
            faces=faces,
            emotions=emotions
        )

        z_loc, z_scale = (
            torch.zeros([1, batch_size, self._latent_space_dim]),
            torch.ones([1, batch_size, self._latent_space_dim])
        )
        
        if self.use_cuda:
            z_loc = z_loc.cuda()
            z_scale = z_scale.cuda()

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

        return z_loc_expert, z_scale_expert, z_loc, z_scale

    def sample_latent(self, z_loc: torch.Tensor, z_scale: torch.Tensor) -> torch.Tensor:
        """
        TODO: this is duplicate... extract
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        """
        epsilon: torch.Tensor = torch.randn_like(z_loc)
        latent_sample: torch.Tensor = z_loc + epsilon * z_scale

        return latent_sample

    def generate(self, latent_sample: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        face_reconstruction = self._face_decoder.forward(latent_sample)
        emotion_reconstruction = self._emotion_decoder.forward(latent_sample)

        return face_reconstruction, emotion_reconstruction

    def forward(self, faces=None, emotions=None, sample=True) -> Tuple[torch.Tensor, ...]:
        # Infer the latent distribution parameters
        z_loc_expert, z_scale_expert, _, _ = self.infer_latent(
            faces=faces,
            emotions=emotions
        )
        
        # Sample from the latent space 
        if sample:
            latent_sample: torch.Tensor = self.sample_latent(
                z_loc=z_loc_expert,
                z_scale=z_scale_expert
            )
        else:
            latent_sample: torch.Tensor = z_loc_expert

        # Reconstruct inputs based on that Gaussian sample
        face_reconstruction, emotions_reconstruction = self.generate(
            latent_sample=latent_sample
        )

        return face_reconstruction, emotions_reconstruction, z_loc_expert, z_scale_expert