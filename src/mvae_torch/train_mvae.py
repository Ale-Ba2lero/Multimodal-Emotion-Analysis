import logging
import os
from typing import Tuple, List, Generator

#import hydra
import numpy
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch_mvae_util
import multimodal_vae  
import nn_modules
from config_args import ConfigModelArgs, ConfigTrainArgs


def build_model(
    cat_dim: int,
    latent_space_dim: int,
    hidden_dim: int,
    loss_weights: dict,
    expert_type: str,
    use_cuda: bool
) -> torch.nn.Module:
    # TODO: add support for loading a pretrained model

    # Build the face modality components
    face_encoder: torch.nn.Module = nn_modules.ImageEncoder(
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim
    )
    face_decoder: torch.nn.Module = nn_modules.ImageDecoder(
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim
    )

    # Build the discrete emotion category modality components
    emocat_encoder: torch.nn.Module = nn_modules.EmotionEncoder(
        input_dim=cat_dim,
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim,
        use_cuda=use_cuda
    )
    emocat_decoder: torch.nn.Module = nn_modules.EmotionDecoder(
        output_dim=cat_dim,
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim,
        use_cuda=use_cuda
    )

    # Create the expert
    if expert_type == "poe":
        # Should the epsilon be parameterized?
        expert: torch_mvae_util.Expert = torch_mvae_util.ProductOfExperts()
    elif expert_type == "moe":
        expert: torch_mvae_util.Expert = torch_mvae_util.MixtureOfExpertsComparableComplexity()
    else:
        raise ValueError(f"Unknown expert type '{expert_type}'")

    # Build the model
    mvae: torch.nn.Module = multimodal_vae.MultimodalVariationalAutoencoder(
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
        emotions
) -> dict:
    # Zero the parameter gradients
    optimizer.zero_grad()

    (
        face_reconstruction,
        emotion_reconstruction,
        z_loc_expert,
        z_scale_expert
    ) = model(
        faces=faces, emotions=emotions
    )

    loss = model.loss_function(
        faces=faces,
        emotions=emotions,
        faces_reconstruction=face_reconstruction,
        emotions_reconstruction=emotion_reconstruction,
        z_loc=z_loc_expert,
        z_scale=z_scale_expert,
        beta=beta
    )

    loss["total_loss"].backward()
    optimizer.step()

    return loss


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
        cfg: ConfigTrainArgs
) -> None:
    checkpoint_every=None
    save_model=False
    
    torch.manual_seed(seed=seed)

    # Setup the optimizer
    adam_args = {"lr": learning_rate, "betas": optim_betas}
    optimizer = torch.optim.Adam(params=mvae_model.parameters(), **adam_args)

    annealing_beta_gen_factory = torch_mvae_util.AnnealingBetaGeneratorFactory(
        annealing_type=cfg.annealing_type,
        training_config=cfg
    )
    annealing_beta_generator: Generator[float, None, None] = annealing_beta_gen_factory.get_annealing_beta_generator(
        num_iterations=num_epochs
    )

    training_losses: dict = {'total_loss': [], 'multimodal_loss':[], 'faces_loss':[], 'emotions_loss':[]}
    # Training loop
    for epoch_num in tqdm(range(num_epochs)):
        # Initialize loss accumulator and the progress bar
        epoch_losses: dict = {'total_loss': [], 'multimodal_loss':[], 'faces_loss':[], 'emotions_loss':[]}
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
            
            losses: dict = eval_model_training(
                model=mvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                emotions=emotions
            )
            multimodal_loss = float(losses["total_loss"].cpu().detach().numpy())
            epoch_losses['multimodal_loss'].append(multimodal_loss)
            
            losses: dict = eval_model_training(
                model=mvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                emotions=None
            )
            faces_loss = float(losses["total_loss"].cpu().detach().numpy())
            epoch_losses['faces_loss'].append(faces_loss)
            losses: dict = eval_model_training(
                model=mvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=None,
                emotions=emotions
            )
            emotions_loss = float(losses["total_loss"].cpu().detach().numpy())
            epoch_losses['emotions_loss'].append(emotions_loss)
            
            epoch_losses['total_loss'].append(multimodal_loss)
            epoch_losses['total_loss'].append(faces_loss)
            epoch_losses['total_loss'].append(emotions_loss)
            
        training_losses['total_loss'].append(numpy.nanmean(epoch_losses['total_loss']))
        training_losses['multimodal_loss'].append(numpy.nanmean(epoch_losses['multimodal_loss']))
        training_losses['faces_loss'].append(numpy.nanmean(epoch_losses['faces_loss']))
        training_losses['emotions_loss'].append(numpy.nanmean(epoch_losses['emotions_loss']))
        
        # Report training diagnostics -  TRIMODAL
        print(
            f"Mean total loss: {training_losses['total_loss'][-1]:.5};\n"
            f"Mean all modalities loss: {training_losses['multimodal_loss'][-1]:.5};\n"
            f"Mean faces loss: {training_losses['faces_loss'][-1]:.5};\n"
            f"Mean emotions loss: {training_losses['emotions_loss'][-1]:.5};\n"
        )

        if checkpoint_every is not None:
            if (epoch_num + 1) % checkpoint_every == 0:
                checkpoint_save_path: str = os.path.join(checkpoint_path, f"ext_mmvae_epoch_{epoch_num + 1:03}.pt")
                torch.save(mvae_model.state_dict(), checkpoint_save_path)
                logger.info(f"Saved checkpoint to {checkpoint_save_path}")

    if save_model:
        # Do a global and a local save of the model (local to Hydra outputs)
        torch.save(mvae_model.state_dict(), cfg.train.plain.model_save_path)
        torch.save(mvae_model.state_dict(), "ravdess_mmvae_pretrained.pt")
        logger.info(f"Saved model to '{cfg.train.plain.model_save_path}', and also locally.")

        # Do a global and local save of the training stats (local to Hydra outputs)
        torch.save(training_losses, cfg.train.plain.stats_save_path)
        torch.save(training_losses, "ravdess_mmvae_pretrained_stats.pt")
        logger.info(f"Saved model to '{cfg.train.plain.stats_save_path}', and also locally.")
        
    return training_losses
