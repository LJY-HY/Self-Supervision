import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from utils.utils import *
from dataset.CIFAR.CIFAR import CIFAR10_noisy,CIFAR100_noisy

def cifar10(args):
    TF_list = get_transform(args)
    if args.mode in ['SupCon','SimCLR']:
        train_proj_dataset = CIFAR10_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, corruption_prob = args.proj_noise_rate, transform = TF_list[0], download=True)
        train_linear_dataset = CIFAR10_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, corruption_prob = args.linear_noise_rate, transform = TF_list[1], download=True)
        test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=False, transform = TF_list[-1], download=False)
        if args.linear_noise_rate == args.proj_noise_rate:
            # keep randomness
            train_linear_dataset.train_labels = train_proj_dataset.train_labels
            train_linear_dataset.train_labels_true = train_proj_dataset.train_labels_true
        train_proj_dataloader = DataLoader(train_proj_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        train_linear_dataloader = DataLoader(train_linear_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return train_proj_dataloader, train_linear_dataloader, test_dataloader
    elif args.mode in ['Xent']:
        train_dataset = CIFAR10_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, corruption_prob = args.noise_rate, transform = TF_list[1], download=True)
        test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=False, transform = TF_list[-1], download=False)
        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return train_dataloader, test_dataloader
    
def cifar100(args):
    TF_list = get_transform(args)
    if args.mode in ['SupCon','SimCLR']:
        train_proj_dataset = CIFAR100_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, corruption_prob = args.proj_noise_rate, transform = TF_list[0], download=True)
        train_linear_dataset = CIFAR100_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, corruption_prob = args.linear_noise_rate, transform = TF_list[1], download=True)
        test_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=False, transform = TF_list[-1], download=False)
        if args.linear_noise_rate == args.proj_noise_rate:
            # keep randomness
            train_linear_dataset.train_labels = train_proj_dataset.train_labels
            train_linear_dataset.train_labels_true = train_proj_dataset.train_labels_true
        train_proj_dataloader = DataLoader(train_proj_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        train_linear_dataloader = DataLoader(train_linear_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return train_proj_dataloader, train_linear_dataloader, test_dataloader
    elif args.mode in ['Xent']:
        train_dataset = CIFAR100_noisy(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, corruption_prob = args.noise_rate, transform = TF_list[1], download=True)
        test_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=False, transform = TF_list[-1], download=False)
        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return train_dataloader, test_dataloader