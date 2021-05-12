import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from utils.utils import *
from dataset.CIFAR.CIFAR import CIFAR10_noisy,CIFAR100_noisy

def cifar10(args, mode = 'train'):
    train_proj_TF,train_linear_TF, test_TF = get_transform(args,mode)

    if args.linear_noise_rate == args.proj_noise_rate :
        train_proj_dataset = CIFAR10_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, corruption_prob = args.proj_noise_rate, transform = train_proj_TF, download=True)
        train_linear_dataset = CIFAR10_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, corruption_prob = args.linear_noise_rate, transform = train_linear_TF, download=True)
        # keep randomness
        train_linear_dataset.train_labels = train_proj_dataset.train_labels
        train_linear_dataset.train_labels_true = train_proj_dataset.train_labels_true
    elif args.linear_noise_rate != args.proj_noise_rate:
        train_linear_dataset = CIFAR10_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, corruption_prob = args.linear_noise_rate, transform = train_proj_TF, download=True)
        train_proj_dataset = CIFAR10_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, corruption_prob = args.proj_noise_rate, transform = train_linear_TF, download=True)
    test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=False, transform = test_TF, download=False)

    train_proj_dataloader = DataLoader(train_proj_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    train_linear_dataloader = DataLoader(train_linear_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    return train_proj_dataloader, train_linear_dataloader, test_dataloader

def cifar100(args, mode = 'train'):
    train_proj_TF,train_linear_TF, test_TF = get_transform(args,mode)

    if args.linear_noise_rate == args.proj_noise_rate :
        train_proj_dataset = CIFAR100_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, corruption_prob = args.proj_noise_rate, transform = train_proj_TF, download=True)
        train_linear_dataset = CIFAR100_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, corruption_prob = args.linear_noise_rate, transform = train_linear_TF, download=True)
        # keep randomness
        train_linear_dataset.self.train_labels = train_proj_dataset.train_labels
        train_linear_dataset.self.train_labels_true = train_proj_dataset.train_labels_true
    elif args.linear_noise_rate != args.proj_noise_rate:
        train_linear_dataset = CIFAR100_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, corruption_prob = args.linear_noise_rate, transform = train_proj_TF, download=True)
        train_proj_dataset = CIFAR100_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, corruption_prob = args.proj_noise_rate, transform = train_linear_TF, download=True)
    test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=False, transform = test_TF, download=False)

    train_proj_dataloader = DataLoader(train_proj_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    train_linear_dataloader = DataLoader(train_linear_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    return train_proj_dataloader, train_linear_dataloader, test_dataloader