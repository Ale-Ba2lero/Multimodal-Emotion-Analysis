import logging
from typing import Tuple

import torch
import torch.nn.functional as torch_functional

from util import Expert


class ExteroceptiveMultimodalVariationalAutoencoder(torch.nn.Module):

    def __init__(
            self,
            face_encoder: torch.nn.Module,
            face_decoder: torch.nn.Module,
            voice_encoder: torch.nn.Module,
            voice_decoder: torch.nn.Module,
            va_encoder: torch.nn.Module,
            va_decoder: torch.nn.Module,
            loss_weights: dict,
            expert: Expert,
            latent_space_dim: int,
            use_cuda: bool = False
    ) -> None:
        super(ExteroceptiveMultimodalVariationalAutoencoder, self).__init__()
        self._logger = logging.getLogger(ExteroceptiveMultimodalVariationalAutoencoder.__name__)

        self._face_encoder: torch.nn.Module = face_encoder
        self._face_decoder: torch.nn.Module = face_decoder
        self._voice_encoder: torch.nn.Module = voice_encoder
        self._voice_decoder: torch.nn.Module = voice_decoder
        self._va_encoder: torch.nn.Module = va_encoder
        self._va_decoder: torch.nn.Module = va_decoder

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
            voices: torch.Tensor,
            annotations: torch.Tensor,
            faces_reconstruction: torch.Tensor,
            voices_reconstruction: torch.Tensor,
            annotations_reconstruction: torch.Tensor,
            z_loc: torch.Tensor,
            z_scale: torch.Tensor,
            beta: float
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
        if voices is not None:
            voices_reconstruction_loss: torch.Tensor = torch_functional.mse_loss(voices_reconstruction, voices)
        else:
            voices_reconstruction_loss: torch.Tensor = torch.Tensor([0.0])
        if annotations is not None:
            annotations_reconstruction_loss: torch.Tensor = torch_functional.mse_loss(
                annotations_reconstruction, annotations
            )
        else:
            annotations_reconstruction_loss: torch.Tensor = torch.Tensor([0.0])

        if self.use_cuda:
            faces_reconstruction_loss = faces_reconstruction_loss.cuda()
            voices_reconstruction_loss = voices_reconstruction_loss.cuda()
            annotations_reconstruction_loss = annotations_reconstruction_loss.cuda()

        faces_reconstruction_loss = self._loss_weights["face"] * faces_reconstruction_loss
        voices_reconstruction_loss = self._loss_weights["voice"] * voices_reconstruction_loss
        annotations_reconstruction_loss = self._loss_weights["va"] * annotations_reconstruction_loss

        reconstruction_loss = faces_reconstruction_loss + voices_reconstruction_loss + annotations_reconstruction_loss

        # Calculate the KLD loss
        log_var = torch.log(torch.square(z_scale))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - z_loc ** 2 - log_var.exp(), dim=1), dim=0)

        total_loss = reconstruction_loss + beta * kld_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kld_loss": kld_loss,
            "faces_reconstruction_loss": faces_reconstruction_loss,
            "voices_reconstruction_loss": voices_reconstruction_loss,
            "annotations_reconstruction_loss": annotations_reconstruction_loss
        }

    def _extract_batch_size_from_data(
            self,
            faces: torch.Tensor = None,
            voices: torch.Tensor = None,
            annotations: torch.Tensor = None
    ) -> int:
        if faces is not None:
            batch_size = faces.shape[0]
        elif voices is not None:
            batch_size = voices.shape[0]
        elif annotations is not None:
            batch_size = annotations.shape[0]
        else:
            batch_size = 0

        return batch_size

    def infer_latent(
            self,
            faces: torch.Tensor,
            voices: torch.Tensor,
            annotations: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Use the encoders to get the parameters used to define q(z|x).
        # Initialize the prior expert.
        # We initialize an additional dimension, along which we concatenate all the different experts.
        #   self.experts() then combines the information from these different modalities by multiplying
        #   the Gaussians together.
        batch_size: int = self._extract_batch_size_from_data(
            faces=faces,
            voices=voices,
            annotations=annotations
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

        if voices is not None:
            voice_z_loc, voice_z_scale = self._voice_encoder.forward(voices)

            z_loc = torch.cat((z_loc, voice_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, voice_z_scale.unsqueeze(0)), dim=0)

        if annotations is not None:
            va_z_loc, va_z_scale = self._va_encoder.forward(annotations)

            z_loc = torch.cat((z_loc, va_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, va_z_scale.unsqueeze(0)), dim=0)

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
        audio_reconstruction = self._voice_decoder.forward(latent_sample)
        annotation_reconstruction = self._va_decoder.forward(latent_sample)

        return face_reconstruction, audio_reconstruction, annotation_reconstruction

    def forward(self, faces=None, voices=None, annotations=None) -> Tuple[torch.Tensor, ...]:
        # Infer the latent distribution parameters
        z_loc_expert, z_scale_expert, _, _ = self.infer_latent(
            faces=faces,
            voices=voices,
            annotations=annotations,
        )

        # Sample from the latent space
        latent_sample: torch.Tensor = self.sample_latent(
            z_loc=z_loc_expert,
            z_scale=z_scale_expert
        )

        # Reconstruct inputs based on that Gaussian sample
        face_reconstruction, audio_reconstruction, annotation_reconstruction = self.generate(
            latent_sample=latent_sample
        )

        return face_reconstruction, audio_reconstruction, annotation_reconstruction, z_loc_expert, z_scale_expert
