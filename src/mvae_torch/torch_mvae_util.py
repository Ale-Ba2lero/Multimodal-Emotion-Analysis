"""
Torch utilities
"""

import itertools
import pickle
from typing import Tuple, Generator, List, Any, Optional

#import librosa
import numpy

import numpy.random
import omegaconf
import torch
import tqdm

from torch.utils.data import Dataset

from config_args import ConfigTrainArgs

#from dataset.iemocap.iemocap_enrich import LoadUtils

class AnnealingBetaGenerator(object):
    @staticmethod
    def linear_annealing_beta_generator(
            min_beta: float,
            max_beta: float,
            num_epochs: int
    ) -> Generator[float, None, None]:
        for epoch in range(num_epochs):
            proportion: float = (epoch + 1) / num_epochs
            yield min_beta + proportion * (max_beta - min_beta)

    @staticmethod
    def cyclical_annealing_beta_generator(
            num_iterations: int,
            min_beta: float,
            max_beta: float,
            num_cycles: int,
            annealing_percentage: float
    ) -> Generator[float, None, None]:
        period = int(num_iterations / num_cycles)
        # Linear schedule
        step = (max_beta - min_beta) / (period * annealing_percentage)

        beta = min_beta
        for idx in range(num_iterations):
            yield min(beta, max_beta)

            if idx % period == 0:
                beta = min_beta
            else:
                beta += step

    @staticmethod
    def static_annealing_beta_generator(
            beta: float
    ) -> Generator[float, None, None]:
        while True:
            yield beta


class AnnealingBetaGeneratorFactory(object):
    def __init__(self, annealing_type: str, training_config: ConfigTrainArgs):
        self._annealing_type: str = annealing_type
        self._training_config: ConfigTrainArgs = training_config

    def get_annealing_beta_generator(self, num_iterations: int) -> Generator[float, None, None]:
        if self._annealing_type == "cyclical":
            return AnnealingBetaGenerator.cyclical_annealing_beta_generator(
                num_iterations=num_iterations,
                min_beta=self._training_config.cyclical_annealing["min_beta"],
                max_beta=self._training_config.cyclical_annealing["max_beta"],
                num_cycles=self._training_config.cyclical_annealing["num_cycles"],
                annealing_percentage=self._training_config.cyclical_annealing["annealing_percentage"],
            )
        elif self._annealing_type == "linear":
            return AnnealingBetaGenerator.linear_annealing_beta_generator(
                min_beta=self._training_config.linear_annealing["min_beta"],
                max_beta=self._training_config.linear_annealing["max_beta"],
                num_epochs=num_iterations
            )
        elif self._annealing_type == "static":
            return AnnealingBetaGenerator.static_annealing_beta_generator(
                beta=self._training_config.static_annealing_beta
            )
        else:
            raise ValueError(f"Invalid annealing tyoe '{self._annealing_type}'")

    @property
    def annealing_type(self) -> str:
        return self._annealing_type

    @property
    def training_config(self) -> omegaconf.DictConfig:
        return self._training_config


def all_subsets(elements: List[Any]):
    return itertools.chain(
        *map(lambda x: itertools.combinations(elements, x), range(0, len(elements) + 1))
    )


def all_subsets_except_empty(elements: List[Any]):
    return filter(
        lambda x: len(x) > 0,
        all_subsets(elements)
    )


class Expert(torch.nn.Module):
    def __init__(self):
        super(Expert, self).__init__()


class MixtureOfExpertsComparableComplexity(Expert):
    """
    The MoE could be extended by different weighting the modalities differently
    """
    def __init__(self):
        super(MixtureOfExpertsComparableComplexity, self).__init__()

    def forward(self, loc: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        num_modalities: int = loc.shape[0]

        moe_loc: torch.Tensor = torch.sum(loc, dim=0) / num_modalities
        moe_scale: torch.Tensor = torch.sum(scale, dim=0) / (num_modalities ** 2)

        return moe_loc, moe_scale


class ProductOfExperts(Expert):
    """
    Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param loc: M x D for M experts
    @param scale: M x D for M experts
    """

    def __init__(self, num_const: float = 1e-6) -> None:
        super(ProductOfExperts, self).__init__()

        # Constant for numerical stability (e.g. in division)
        self._eps: float = num_const

    def forward(self, loc: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        scale += self._eps
        # Precision of i-th Gaussian expert (T = 1/sigma^2)
        precision = 1. / scale

        product_loc: torch.Tensor = torch.sum(loc * precision, dim=0) / torch.sum(precision, dim=0)
        product_scale: torch.Tensor = 1. / torch.sum(precision, dim=0)

        return product_loc, product_scale


def split_list_randomly(list_obj: List[Any], split_percentage: float) -> Tuple[List[Any], List[Any]]:
    assert 0 <= split_percentage <= 1
    assert len(list_obj) > 0

    first_part_len: int = int(len(list_obj) * split_percentage)
    first_part_random_elements: List[int] = numpy.random.choice(range(len(list_obj)), size=first_part_len)
    second_part_random_elements: List[int] = [
        idx for idx in range(len(list_obj))
        if idx not in first_part_random_elements
    ]

    first_list: List[Any] = [
        list_obj[idx] for idx in first_part_random_elements
    ]
    second_list: List[Any] = [
        list_obj[idx] for idx in second_part_random_elements
    ]

    return first_list, second_list


def load_preprocessed_dataset(cfg: dict) -> Tuple[Dataset, Dataset]:
    if cfg.model_type == "plain":
        with open(cfg.paths.plain.training_data_path, "rb") as in_stream:
            ravdess_exteroceptive_dataset_train: torch.utils.data.Dataset = pickle.load(in_stream)
        with open(cfg.paths.plain.testing_data_path, "rb") as in_stream:
            ravdess_exteroceptive_dataset_test: torch.utils.data.Dataset = pickle.load(in_stream)
    elif cfg.model_type == "cnn":
        with open(cfg.paths.cnn.training_data_path, "rb") as in_stream:
            ravdess_exteroceptive_dataset_train: torch.utils.data.Dataset = pickle.load(in_stream)
        with open(cfg.paths.cnn.testing_data_path, "rb") as in_stream:
            ravdess_exteroceptive_dataset_test: torch.utils.data.Dataset = pickle.load(in_stream)
    elif cfg.model_type == "vrnn":
        with open(cfg.paths.vrnn.training_data_path, "rb") as in_stream:
            ravdess_exteroceptive_dataset_train: torch.utils.data.Dataset = pickle.load(in_stream)
        with open(cfg.paths.vrnn.testing_data_path, "rb") as in_stream:
            ravdess_exteroceptive_dataset_test: torch.utils.data.Dataset = pickle.load(in_stream)
    else:
        raise ValueError(f"Unknown model type '{cfg.model_type}'")

    return ravdess_exteroceptive_dataset_train, ravdess_exteroceptive_dataset_test


def extract_audio_features(
        file_path: str, desired_sample_rate: float, native_sample_rate: int, win_size: float
) -> List[numpy.ndarray]:
    """
    # TODO: move to feature extraction
    Reads a wav file and stores 3 types of audio features:
        mel-frequency cepstral coefficients,
        mel-scaled spectrogram,
        chromagram.

    :param file_path:
    :param desired_sample_rate: in Hz
    :param native_sample_rate: in Hz
    :param win_size: in seconds
    :return: a list containig the following audio features (in this order):
        mel-frequency cepstral coefficients, mel-scaled spectrogram ant chromagram.
    """
    # Load the data
    #   sr=None would actually preserve the native sample rate, but by providing the argument we make sure
    #   that it won't be different. There are probably better ways to do this.
    data, sample_rate = librosa.load(file_path, sr=native_sample_rate)
    # Calculate the number of samples until next window
    hop_length = int(numpy.round(native_sample_rate / desired_sample_rate))
    n_fft: int = int(numpy.round(native_sample_rate * win_size))

    # mel_spectrogram = numpy.transpose(librosa.feature.melspectrogram(
    #     y=data, sr=native_sample_rate, hop_length=hop_length, n_fft=n_fft
    # ))
    spectral_centroid = numpy.transpose(librosa.feature.spectral_centroid(
        y=data, sr=native_sample_rate, hop_length=hop_length, n_fft=n_fft
    ))
    chromagram = numpy.transpose(librosa.feature.chroma_stft(
        y=data, sr=native_sample_rate, hop_length=hop_length, n_fft=n_fft
    ))
    mfcc = numpy.transpose(librosa.feature.mfcc(
        y=data, sr=native_sample_rate, hop_length=hop_length, n_fft=n_fft
    ))

    return [chromagram, spectral_centroid, mfcc]


def iemocap_add_audio_features(iem_ref: List[dict], desired_sample_rate: int, native_sample_rate: int, win_size: float):
    for utterance in tqdm.tqdm(LoadUtils.iter_over_utterances(iem_ref)):
        utterance["wav"]["wav_features"] = extract_audio_features(
            file_path=utterance["wav"]["wav_path"],
            desired_sample_rate=desired_sample_rate,
            native_sample_rate=native_sample_rate,
            win_size=win_size
        )