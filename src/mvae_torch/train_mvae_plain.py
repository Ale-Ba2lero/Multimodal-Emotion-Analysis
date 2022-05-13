import logging
import os
from typing import Tuple, List, Generator

import hydra
import numpy
import torch
import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

import model.util as util
from model import multimodal_vae, model_component

_logger = logging.getLogger(__name__)


def build_model(
        input_dims: dict,
        latent_space_dim: int,
        va_mlp_hidden_dim: int,
        loss_weights: dict,
        expert_type: str,
        use_cuda: bool
) -> torch.nn.Module:
    # TODO: add support for loading a pretrained model

    # Build the face modality components
    face_encoder: torch.nn.Module = model_component.MultilayerPerceptronEncoder(
        input_dim=input_dims["face"],
        hidden_dim=va_mlp_hidden_dim,
        z_dim=latent_space_dim
    )
    face_decoder: torch.nn.Module = model_component.MultilayerPerceptronDecoderSigmoid(
        input_dim=input_dims["face"],
        hidden_dim=va_mlp_hidden_dim,
        z_dim=latent_space_dim
    )

    # Build the voice modality components
    voice_encoder: torch.nn.Module = model_component.MultilayerPerceptronEncoder(
        input_dim=input_dims["audio"],
        hidden_dim=va_mlp_hidden_dim,
        z_dim=latent_space_dim
    )
    voice_decoder: torch.nn.Module = model_component.MultilayerPerceptronDecoder(
        input_dim=input_dims["audio"],
        hidden_dim=va_mlp_hidden_dim,
        z_dim=latent_space_dim
    )

    # Build the valence-arousal modality components
    va_encoder: torch.nn.Module = model_component.MultilayerPerceptronEncoder(
        input_dim=input_dims["annotation"],
        hidden_dim=va_mlp_hidden_dim,
        z_dim=latent_space_dim
    )
    va_decoder: torch.nn.Module = model_component.MultilayerPerceptronDecoder(
        input_dim=input_dims["annotation"],
        hidden_dim=va_mlp_hidden_dim,
        z_dim=latent_space_dim
    )

    # Create the expert
    if expert_type == "poe":
        # Should the epsilon be parameterized?
        expert: util.Expert = util.ProductOfExperts()
    elif expert_type == "moe":
        expert: util.Expert = util.MixtureOfExpertsComparableComplexity()
    else:
        raise ValueError(f"Unknown expert type '{expert_type}'")

    # Build the model
    exteroceptive_mmvae: torch.nn.Module = mmvae.ExteroceptiveMultimodalVariationalAutoencoder(
        face_encoder=face_encoder,
        face_decoder=face_decoder,
        voice_encoder=voice_encoder,
        voice_decoder=voice_decoder,
        va_encoder=va_encoder,
        va_decoder=va_decoder,
        loss_weights=loss_weights,
        expert=expert,
        latent_space_dim=latent_space_dim,
        use_cuda=use_cuda
    )

    return exteroceptive_mmvae


def eval_model_training(
        model,
        optimizer,
        beta,
        faces,
        voices,
        annotations
) -> dict:
    # Zero the parameter gradients
    optimizer.zero_grad()

    (
        face_reconstruction,
        audio_reconstruction,
        annotation_reconstruction,
        z_loc_expert,
        z_scale_expert
    ) = model(
        faces=faces, voices=voices, annotations=annotations
    )

    loss = model.loss_function(
        faces=faces,
        voices=voices,
        annotations=annotations,
        faces_reconstruction=face_reconstruction,
        voices_reconstruction=audio_reconstruction,
        annotations_reconstruction=annotation_reconstruction,
        z_loc=z_loc_expert,
        z_scale=z_scale_expert,
        beta=beta
    )

    loss["total_loss"].backward()
    optimizer.step()

    return loss


def train(
        mmvae_model: torch.nn.Module,
        dataset: Dataset,
        learning_rate: float,
        optim_betas: Tuple[float, float],
        num_epochs: int,
        batch_size: int,
        checkpoint_every: int,
        checkpoint_path: str,
        save_model: bool,
        seed: int,
        use_cuda: bool,
        cfg: DictConfig
) -> None:
    logger = logging.getLogger(train.__name__)

    logger.info("Currently using Torch version: " + torch.__version__)
    torch.manual_seed(seed=seed)

    # Setup the optimizer
    adam_args = {"lr": learning_rate, "betas": optim_betas}
    optimizer = torch.optim.Adam(params=mmvae_model.parameters(), **adam_args)

    # Create data loader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    annealing_beta_gen_factory = util.AnnealingBetaGeneratorFactory(
        annealing_type=cfg.train.plain.annealing_type,
        training_config=cfg.train.plain
    )
    annealing_beta_generator: Generator[float, None, None] = annealing_beta_gen_factory.get_annealing_beta_generator(
        num_iterations=num_epochs
    )

    training_losses: List[List[float]] = list()
    # Training loop
    for epoch_num in range(num_epochs):
        # Initialize loss accumulator and the progress bar
        epoch_losses: List[List[float]] = list()
        progress_bar = tqdm.tqdm(data_loader)
        annealing_beta = next(annealing_beta_generator)

        logger.info(f"Starting epoch {epoch_num + 1}. Annealing beta {annealing_beta:.2}")

        # Do a training epoch over each mini-batch returned
        #   by the data loader
        for faces, voices, annotations in progress_bar:
            if len(epoch_losses) > 0:
                batch_loss_mean = numpy.nanmean(epoch_losses[-1])
                progress_bar.set_description(f"Epoch {epoch_num + 1:3}; Batch loss: {batch_loss_mean:3.5}")
            else:
                progress_bar.set_description(f"Epoch {epoch_num + 1:3}; Batch loss: Nan")

            # If on GPU put the mini-batch into CUDA memory
            if use_cuda:
                if faces is not None:
                    faces = faces.cuda()
                if voices is not None:
                    voices = voices.cuda()
                if annotations is not None:
                    annotations = annotations.cuda()

            batch_losses = list()

            losses: dict = eval_model_training(
                model=mmvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                voices=voices,
                annotations=annotations
            )
            batch_losses.append(float(losses["total_loss"].cpu().detach().numpy()))

            losses: dict = eval_model_training(
                model=mmvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                voices=voices,
                annotations=None
            )
            batch_losses.append(float(losses["total_loss"].cpu().detach().numpy()))

            losses: dict = eval_model_training(
                model=mmvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                voices=None,
                annotations=annotations
            )
            batch_losses.append(float(losses["total_loss"].cpu().detach().numpy()))

            losses: dict = eval_model_training(
                model=mmvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=None,
                voices=voices,
                annotations=annotations
            )
            batch_losses.append(float(losses["total_loss"].cpu().detach().numpy()))

            losses: dict = eval_model_training(
                model=mmvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=faces,
                voices=None,
                annotations=None
            )
            batch_losses.append(float(losses["total_loss"].cpu().detach().numpy()))

            losses: dict = eval_model_training(
                model=mmvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=None,
                voices=voices,
                annotations=None
            )
            batch_losses.append(float(losses["total_loss"].cpu().detach().numpy()))

            losses: dict = eval_model_training(
                model=mmvae_model,
                optimizer=optimizer,
                beta=annealing_beta,
                faces=None,
                voices=None,
                annotations=annotations
            )
            batch_losses.append(float(losses["total_loss"].cpu().detach().numpy()))

            epoch_losses.append(batch_losses)

        epoch_losses: numpy.ndarray = numpy.array(epoch_losses)
        epoch_losses_means: List[float] = [
            float(numpy.nanmean(epoch_losses)),
            float(numpy.nanmean(epoch_losses[:, 0])),
            float(numpy.nanmean(epoch_losses[:, 1:4])),
            float(numpy.nanmean(epoch_losses[:, 4])),
            float(numpy.nanmean(epoch_losses[:, 5])),
            float(numpy.nanmean(epoch_losses[:, 6]))
        ]
        training_losses.append(epoch_losses_means)
        # Report training diagnostics -  TRIMODAL
        print(
            f"Mean epoch loss: {epoch_losses_means[0]:.5}; "
            f"Mean all modalities loss: {epoch_losses_means[1]:.5}; "
            f"Mean bimodal loss: {epoch_losses_means[2]:.5}; "
            f"Mean faces loss: {epoch_losses_means[3]:.5}; "
            f"Mean voices loss: {epoch_losses_means[4]:.5}; "
            f"Mean annotations loss: {epoch_losses_means[5]:.5}"
        )

        if checkpoint_every is not None:
            if (epoch_num + 1) % checkpoint_every == 0:
                checkpoint_save_path: str = os.path.join(checkpoint_path, f"ext_mmvae_epoch_{epoch_num + 1:03}.pt")
                torch.save(mmvae_model.state_dict(), checkpoint_save_path)
                logger.info(f"Saved checkpoint to {checkpoint_save_path}")

    if save_model:
        # Do a global and a local save of the model (local to Hydra outputs)
        torch.save(mmvae_model.state_dict(), cfg.train.plain.model_save_path)
        torch.save(mmvae_model.state_dict(), "ravdess_mmvae_pretrained.pt")
        logger.info(f"Saved model to '{cfg.train.plain.model_save_path}', and also locally.")

        # Do a global and local save of the training stats (local to Hydra outputs)
        torch.save(training_losses, cfg.train.plain.stats_save_path)
        torch.save(training_losses, "ravdess_mmvae_pretrained_stats.pt")
        logger.info(f"Saved model to '{cfg.train.plain.stats_save_path}', and also locally.")


@hydra.main(config_path="../../../config", config_name="ravdess_exteroceptive")
def main(cfg: DictConfig) -> None:
    training_dataset, testing_dataset = util.load_preprocessed_dataset(cfg=cfg)
    input_dims: dict = util.get_modality_input_dimensions_from_data(
        dataset=training_dataset,
        modality="plain"
    )
    model: torch.nn.Module = build_model(
        input_dims=input_dims,
        latent_space_dim=cfg.model.plain.latent_space_dim,
        va_mlp_hidden_dim=cfg.model.plain.va_mlp_hidden_dim,
        loss_weights=cfg.model.plain.loss_weights,
        expert_type=cfg.model.plain.expert_type,
        use_cuda=cfg.train.plain.use_cuda
    )
    train(
        mmvae_model=model,
        dataset=training_dataset,
        learning_rate=cfg.train.plain.learning_rate,
        optim_betas=cfg.train.plain.optim_betas,
        num_epochs=cfg.train.plain.num_epochs,
        batch_size=cfg.train.plain.batch_size,
        checkpoint_every=cfg.train.plain.checkpoint_every,
        checkpoint_path=cfg.train.plain.checkpoint_path,
        save_model=cfg.train.plain.save_model,
        seed=cfg.train.plain.seed,
        use_cuda=cfg.train.plain.use_cuda,
        cfg=cfg
    )


if __name__ == "__main__":
    main()
