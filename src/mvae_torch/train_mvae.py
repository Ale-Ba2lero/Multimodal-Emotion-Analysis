import logging
import os
from typing import Tuple, List, Generator

#import hydra
import numpy
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from multimodal_vae import MultimodalVariationalAutoencoder
import nn_modules as nnm

from torch_mvae_util import Expert, ProductOfExperts, MixtureOfExpertsComparableComplexity, AnnealingBetaGeneratorFactory
from config_args import ConfigTrainArgs


def build_model(
    cat_dim: int,
    latent_space_dim: int,
    hidden_dim: int,
    num_filters: int,
    loss_weights: dict,
    expert_type: str,
    use_cuda: bool
) -> torch.nn.Module:
    # TODO: add support for loading a pretrained model

    face_encoder: torch.nn.Module = nnm.DCGANFaceEncoder(
        z_dim=latent_space_dim,
        num_filters=num_filters
    )
    face_decoder: torch.nn.Module = nnm.DCGANFaceDecoder(
        z_dim=latent_space_dim,
        num_filters=num_filters
    )

    # Build the discrete emotion category modality components
    emocat_encoder: torch.nn.Module = nnm.EmotionEncoder(
        input_dim=cat_dim,
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim,
        use_cuda=use_cuda
    )
    emocat_decoder: torch.nn.Module = nnm.EmotionDecoder(
        output_dim=cat_dim,
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim,
        use_cuda=use_cuda
    )

    # Create the expert
    if expert_type == "poe":
        # Should the epsilon be parameterized?
        expert: Expert = ProductOfExperts(num_const=1e-8)
    elif expert_type == "moe":
        expert: Expert = MixtureOfExpertsComparableComplexity()
    else:
        raise ValueError(f"Unknown expert type '{expert_type}'")

    # Build the model
    mvae: torch.nn.Module = MultimodalVariationalAutoencoder(
        face_encoder=face_encoder,
        face_decoder=face_decoder,
        emotion_encoder=emocat_encoder,
        emotion_decoder=emocat_decoder,
        loss_weights=loss_weights,
        expert=expert,
        latent_space_dim=latent_space_dim,
        use_cuda=use_cuda
    )

    return mvae


def eval_model_training(
        model,
        optimizer,
        beta,
        faces,
        emotions,
        ignore_faces=False,
        ignore_emotions=False
) -> dict:
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    if ignore_faces:
        input_faces = None
    else:
        input_faces = faces
    if ignore_emotions:
        input_emotions = None
    else:
        input_emotions = emotions

    (
        face_reconstruction,
        emotion_reconstruction,
        z_loc_expert,
        z_scale_expert,
        latent_sample
    ) = model(
        faces=input_faces, emotions=input_emotions
    )

    loss = model.loss_function(
        faces=faces,
        emotions=emotions,
        faces_reconstruction=face_reconstruction,
        emotions_reconstruction=emotion_reconstruction,
        z_loc=z_loc_expert,
        z_scale=z_scale_expert,
        beta=beta,
        latent_sample=latent_sample
    )

    loss["total_loss"].backward()
    optimizer.step()

    return loss


class StatLoss:
    def __init__(self):
        self.total_loss = []
        self.reconstruction_loss = []
        #self.kld_loss = []
        self.mmd = []
        self.faces_reconstruction_loss = []
        self.emotions_reconstruction_loss = []
        self.latent_sample = []

def train(
        mvae_model: torch.nn.Module,
        dataset_loader: DataLoader,
        learning_rate: float,
        optim_betas: Tuple[float, float],
        num_epochs: int,
        batch_size: int,
        checkpoint_every: int,
        checkpoint_path: str,
        save_model: bool,
        seed: int,
        use_cuda: bool,
        cfg: ConfigTrainArgs,
        print_loss: bool = False
) -> None:
    checkpoint_every=None
    save_model=False
    
    torch.manual_seed(seed=seed)

    # Setup the optimizer
    adam_args = {"lr": learning_rate, "betas": optim_betas}
    optimizer = torch.optim.Adam(params=mvae_model.parameters(), **adam_args)

    annealing_beta_gen_factory = AnnealingBetaGeneratorFactory(
        annealing_type=cfg.annealing_type,
        training_config=cfg
    )
    annealing_beta_generator: Generator[float, None, None] = annealing_beta_gen_factory.get_annealing_beta_generator(
        num_iterations=num_epochs
    )

    training_losses:dict = {'multimodal_loss': StatLoss(), 'face_loss': StatLoss(), 'emotion_loss': StatLoss()}
    # Training loop
    for epoch_num in tqdm(range(num_epochs)):
        # Initialize loss accumulator and the progress bar
        multimodal_loss = StatLoss()
        face_loss = StatLoss()
        emotion_loss = StatLoss()
        annealing_beta = next(annealing_beta_generator)

        # Do a training epoch over each mini-batch returned
        #   by the data loader
        for sample in dataset_loader:
            faces, emotions = sample['image'], sample['cat']
            # If on GPU put the mini-batch into CUDA memory
            if use_cuda:
                if faces is not None:
                    faces = faces.cuda()
                if emotions is not None:
                    emotions = emotions.cuda()
            
            # multimodal loss
            m_losses: dict = eval_model_training(
                model=mvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                emotions=emotions
            )
                
            multimodal_loss.total_loss.append(float(m_losses["total_loss"].cpu().detach().numpy()))
            multimodal_loss.reconstruction_loss.append(float(m_losses["reconstruction_loss"].cpu().detach().numpy()))
            #multimodal_loss.kld_loss.append(float(m_losses["kld_loss"].cpu().detach().numpy()))
            multimodal_loss.mmd.append(float(m_losses["mmd"].cpu().detach().numpy()))
            multimodal_loss.faces_reconstruction_loss.append(float(m_losses["faces_reconstruction_loss"].cpu().detach().numpy()))
            multimodal_loss.emotions_reconstruction_loss.append(float(m_losses["emotions_reconstruction_loss"].cpu().detach().numpy()))
            
            # face only loss
            f_losses: dict = eval_model_training(
                model=mvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                emotions=None
            )
            
            '''
            f_losses: dict = eval_model_training(
                model=mvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                emotions=emotions,
                ignore_emotions=True
            )
            '''
            face_loss.total_loss.append(float(f_losses["total_loss"].cpu().detach().numpy()))
            face_loss.reconstruction_loss.append(float(f_losses["reconstruction_loss"].cpu().detach().numpy()))
            #face_loss.kld_loss.append(float(f_losses["kld_loss"].cpu().detach().numpy()))
            face_loss.mmd.append(float(f_losses["mmd"].cpu().detach().numpy()))
            face_loss.faces_reconstruction_loss.append(float(f_losses["faces_reconstruction_loss"].cpu().detach().numpy()))
            face_loss.emotions_reconstruction_loss.append(float(f_losses["emotions_reconstruction_loss"].cpu().detach().numpy()))
            
            # emotion only loss
            e_losses: dict = eval_model_training(
                model=mvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=None,
                emotions=emotions
            )
            '''
            e_losses: dict = eval_model_training(
                model=mvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                emotions=emotions,
                ignore_faces=True
            )
            '''
            emotion_loss.total_loss.append(float(e_losses["total_loss"].cpu().detach().numpy()))
            emotion_loss.reconstruction_loss.append(float(e_losses["reconstruction_loss"].cpu().detach().numpy()))
            #emotion_loss.kld_loss.append(float(e_losses["kld_loss"].cpu().detach().numpy()))
            emotion_loss.mmd.append(float(e_losses["mmd"].cpu().detach().numpy()))
            emotion_loss.faces_reconstruction_loss.append(float(e_losses["faces_reconstruction_loss"].cpu().detach().numpy()))
            emotion_loss.emotions_reconstruction_loss.append(float(e_losses["emotions_reconstruction_loss"].cpu().detach().numpy()))
            
        
        training_losses['multimodal_loss'].total_loss.append(numpy.nanmean(multimodal_loss.total_loss))
        training_losses['multimodal_loss'].reconstruction_loss.append(numpy.nanmean(multimodal_loss.reconstruction_loss))
        #training_losses['multimodal_loss'].kld_loss.append(numpy.nanmean(multimodal_loss.kld_loss))
        training_losses['multimodal_loss'].mmd.append(numpy.nanmean(multimodal_loss.mmd))
        training_losses['multimodal_loss'].faces_reconstruction_loss.append(numpy.nanmean(multimodal_loss.faces_reconstruction_loss))
        training_losses['multimodal_loss'].emotions_reconstruction_loss.append(numpy.nanmean(multimodal_loss.emotions_reconstruction_loss))
        
        training_losses['face_loss'].total_loss.append(numpy.nanmean(face_loss.total_loss))
        training_losses['face_loss'].reconstruction_loss.append(numpy.nanmean(face_loss.reconstruction_loss))
        #training_losses['face_loss'].kld_loss.append(numpy.nanmean(face_loss.kld_loss))
        training_losses['face_loss'].mmd.append(numpy.nanmean(face_loss.mmd))
        training_losses['face_loss'].faces_reconstruction_loss.append(numpy.nanmean(face_loss.faces_reconstruction_loss))
        training_losses['face_loss'].emotions_reconstruction_loss.append(numpy.nanmean(face_loss.emotions_reconstruction_loss))
        
        training_losses['emotion_loss'].total_loss.append(numpy.nanmean(emotion_loss.total_loss))
        training_losses['emotion_loss'].reconstruction_loss.append(numpy.nanmean(emotion_loss.reconstruction_loss))
        #training_losses['emotion_loss'].kld_loss.append(numpy.nanmean(emotion_loss.kld_loss))
        training_losses['emotion_loss'].mmd.append(numpy.nanmean(emotion_loss.mmd))
        training_losses['emotion_loss'].faces_reconstruction_loss.append(numpy.nanmean(emotion_loss.faces_reconstruction_loss))
        training_losses['emotion_loss'].emotions_reconstruction_loss.append(numpy.nanmean(emotion_loss.emotions_reconstruction_loss))
                
    return training_losses
