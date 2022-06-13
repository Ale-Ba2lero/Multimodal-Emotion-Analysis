"""
Torch utilities
"""

import itertools
import pickle
from typing import Tuple, Generator, List, Any, Optional
import matplotlib.pyplot as plt
import util.RAVDESS_dataset_util as Rd

import numpy

import numpy.random
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
        scale = scale + self._eps
        # Precision of i-th Gaussian expert (T = 1/sigma^2)
        precision = 1. / scale

        product_loc: torch.Tensor = torch.sum(loc * precision, dim=0) / torch.sum(precision, dim=0)
        product_scale: torch.Tensor = 1. / torch.sum(precision, dim=0)

        return product_loc, product_scale

    
def emotions_to_images(model, img_size=64, use_cuda=True, model_eval=True):
    if model_eval: model.eval()
    else: model.train()
        
    fig, ax = plt.subplots(nrows=2, ncols=4,figsize=[15, 8])
    labels = torch.tensor(list(Rd.emocat.keys()))
    
    if use_cuda: labels = labels.to('cuda')
    
    rec_imgs = []
    
    with torch.no_grad():
        reconstructed_images, _, _, _, _ = model(faces=None, emotions=labels)

    for idx in range(len(labels)):        
        reconstructed_img = reconstructed_images[idx].cpu().view(3, img_size, img_size).detach().numpy()
        reconstructed_img = numpy.array(reconstructed_img*255., dtype='uint8').transpose((1, 2, 0))
        rec_imgs.append(reconstructed_img)
        
    for i, axi in enumerate(ax.flat):
        axi.imshow(rec_imgs[i])
        axi.set_title(Rd.emocat[labels[i].item()], fontsize=20, color="green")
    
    plt.show()
    return rec_imgs


def images_to_images(model, dataset_loader, num_images=4, img_size=64, model_eval=True, use_cuda=True):
    
    if model_eval:
        model.eval()
    else:
        model.train()
        
    sample = next(iter(dataset_loader))
    images = sample['image']
    
    if use_cuda:
        images = images.cuda()
        
    batch_size = images.shape[0]
        
    input_array = numpy.zeros(shape=(img_size, 1, 3), dtype="uint8")
    reconstructed_array = numpy.zeros(shape=(img_size, 1, 3), dtype="uint8")
    
    reconstructed_images, _, _, _, _ = model(faces=images, emotions=None)
    
    if num_images > batch_size: num_images=batch_size
        
    for idx in range(num_images):
        input_image = images[idx]
        
        # storing the input image
        input_image_display = numpy.array(input_image.cpu()*255., dtype='uint8').transpose((1, 2, 0))
        input_array = numpy.concatenate((input_array, input_image_display), axis=1)
        
        # generating the reconstructed image and adding to array
        input_image = input_image.view(1, 3, img_size, img_size)
        
        reconstructed_img = reconstructed_images[idx].cpu().view(3, img_size, img_size).detach().numpy()
        reconstructed_img = numpy.array(reconstructed_img*255., dtype='uint8').transpose((1, 2, 0))
        reconstructed_array = numpy.concatenate((reconstructed_array, reconstructed_img), axis=1)
        
    input_array = input_array[:,1:,:]
    reconstructed_array = reconstructed_array[:,1:,:]
    display_array = numpy.concatenate((input_array, reconstructed_array), axis=0)
    plt.figure(figsize = (40,10))
    plt.imshow(display_array)
    return display_array


def recon_and_classiffication_accuracy(model, dataset_loader, num_samples=100, model_eval = True):
    if model_eval:
        model.eval()
    else:
        model.train()
    
    match = 0
    total = 0
    
    emo_acc = torch.zeros((8,), dtype=torch.int32)
    total_emo = torch.zeros((8,), dtype=torch.int32)
    
    with torch.no_grad():
        for sample in tqdm.tqdm(range(num_samples)):
            random_labels = torch.randint(low=0, high=8, size=(16,)).to("cuda")
            reconstructed_image, _, _, _, _ = model(faces=None, emotions=random_labels)
            _, reconstructed_emotions, _, _, _ = model(faces=reconstructed_image, emotions=None)
            reconstructed_emotions = torch.argmax(reconstructed_emotions, 1)

            for idx in range(len(random_labels)):
                total += 1
                total_emo[random_labels[idx]] += 1
                if random_labels[idx] == reconstructed_emotions[idx]:
                    match += 1
                    emo_acc[random_labels[idx]] += 1
    
    acc = match / total
    emo_acc = emo_acc / total_emo
    return acc, emo_acc


def classiffication_accuracy(model, dataset_loader, model_eval = True):
    if model_eval:
        model.eval()
    else:
        model.train()
        
    match = 0
    total = 0
    
    emo_acc = torch.zeros((8,), dtype=torch.int32)
    total_emo = torch.zeros((8,), dtype=torch.int32)
    
    with torch.no_grad():
        for sample in tqdm.tqdm(iter(dataset_loader)):
            labels = sample['cat'].cuda()
            image = sample['image'].cuda()

            _, reconstructed_emotions, _, _, _ = model(faces=image, emotions=None)
            reconstructed_emotions = torch.argmax(reconstructed_emotions, 1)

            for idx in range(len(labels)):
                total += 1
                total_emo[labels[idx]] += 1
                if labels[idx] == reconstructed_emotions[idx]:
                    match += 1
                    emo_acc[labels[idx]] += 1
    
    acc = match / total
    emo_acc = emo_acc / total_emo
    return acc
    

def print_losses(training_losses, title=None, skipframe=0):
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(14, 10))
    if title is not None:
        fig.suptitle(title,
          fontsize=20,
          color="green")

    ax1.set_title('Reconstruction loss')
    ax1.plot(training_losses['multimodal_loss'].total_loss[skipframe:], color='red', label='multimodal')
    ax1.plot(training_losses['emotion_loss'].total_loss[skipframe:], color='green', label='emotion')
    ax1.plot(training_losses['face_loss'].total_loss[skipframe:], color='blue', label='face')
    ax1.legend(loc="upper right")
    
    ax2.set_title('KLD loss')
    ax2.plot(training_losses['multimodal_loss'].kld_loss[skipframe:], color='red', label='multimodal')
    ax2.plot(training_losses['emotion_loss'].kld_loss[skipframe:], color='green', label='emotion')
    ax2.plot(training_losses['face_loss'].kld_loss[skipframe:], color='blue', label='face')
    ax2.legend(loc="upper right")
    
    ax3.set_title('MMD Loss')
    ax3.plot(training_losses['multimodal_loss'].mmd_loss[skipframe:], color='red', label='multimodal')
    ax3.plot(training_losses['emotion_loss'].mmd_loss[skipframe:], color='green', label='emotion')
    ax3.plot(training_losses['face_loss'].mmd_loss[skipframe:], color='blue', label='face')
    ax3.legend(loc="upper right")

    ax4.set_title('Face reconstruction loss')
    ax4.plot(training_losses['multimodal_loss'].faces_reconstruction_loss[skipframe:], color='red', label='multimodal')
    #ax4.plot(training_losses['emotion_loss'].faces_reconstruction_loss[skipframe:], color='green', label='emotion')
    ax4.plot(training_losses['face_loss'].faces_reconstruction_loss[skipframe:], color='blue', label='face')
    ax4.legend(loc="upper right")

    ax5.set_title('Emotion reconstruction loss')
    ax5.plot(training_losses['multimodal_loss'].emotions_reconstruction_loss[skipframe:], color='red', label='multimodal')
    ax5.plot(training_losses['emotion_loss'].emotions_reconstruction_loss[skipframe:], color='green', label='emotion')
    #ax5.plot(training_losses['face_loss'].emotions_reconstruction_loss[skipframe:], color='blue', label='face')
    ax5.legend(loc="upper right")    

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in fig.get_axes():
    #    ax.label_outer()