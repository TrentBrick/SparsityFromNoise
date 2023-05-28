import copy
from enum import Enum, auto
from easydict import EasyDict
from typing import List
import numpy as np
import torch
from models import *
from torch import nn
from py_scripts import LightningDataModule

# I should be able to set the name and the associated parameters directly. then when I say what I want it auto selects this option. 
# thus it is like an enum where it equals a dictionary. 

dataset_params = EasyDict(

    MNIST = dict(
        img_dim = 28,
        input_size = 784,
        nclasses = 10, 
        nchannels=1, 
        smallest_class_number=892, 
        label_iter = range(10),
        torchified_dataset_suffix='',
        num_data_splits = 5,

    ), 

    CIFAR10 = dict(
        img_dim = 32,
        input_size = 3072,
        nclasses = 10, 
        nchannels=3, 
        smallest_class_number=5000, 
        label_iter = range(10),
        # this suffix is combined with the overall data directory, and if it is a split dataset or not. this is in combine params and then also processed in the data loaders itself. 
        torchified_dataset_suffix = 'CIFAR10/',
        num_data_splits = 5,
        num_data_points = 50000,

    ),

    CIFAR100 = dict(
        img_dim = 32,
        input_size = 3072,
        nclasses = 100, 
        nchannels=3, 
        smallest_class_number=500, 
        label_iter = range(100),
        torchified_dataset_suffix = 'CIFAR100/',
        num_data_splits = 50,
    ), 

    Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10 = dict(
        img_dim = None,
        input_size = 256,
        nclasses = 10, 
        nchannels=1, 
        smallest_class_number=5000, 
        label_iter = range(10), 
        num_data_splits=5,
        torchified_dataset_suffix = 'CachedOutputs/ConvMixerWTransforms_ImgNet32_CIFAR10/',
    ), 

    Cached_ConvMixer_WTransforms_ImageNet32_CIFAR100 = dict(
        img_dim = None,
        input_size = 256,
        nclasses = 100, 
        nchannels=1, 
        smallest_class_number=500, 
        label_iter = range(100), 
        num_data_splits=50,
        torchified_dataset_suffix = 'CachedOutputs/ConvMixerWTransforms_ImgNet32_CIFAR100/',

    ), 

    Cached_ConvMixer_WTransforms_ImageNet32_ImageNet32 = dict(
        img_dim = None,
        input_size = 256,
        nclasses = 1000, 
        nchannels=1, 
        smallest_class_number=5000, 
        label_iter = range(10),
        num_data_splits = 500,
        torchified_dataset_suffix = 'CachedOutputs/ConvMixerWTransforms_ImgNet32_ImageNet32/',
    ), 
)