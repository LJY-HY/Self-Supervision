import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import os.path
from tqdm import tqdm
import argparse
from utils.arguments import get_arguments
from utils.utils import *
from utils.loss import *
from dataset.cifar import *
from torch.utils.tensorboard import SummaryWriter

def main():
    # Argument parsing
    args = argparse.ArgumentParser()
    args = get_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # tensorboard
    writer = SummaryWriter('SupConLoss/'+args.mode+'_proj_'+str(args.proj_noise_rate)+'_linear_'+str(args.linear_noise_rate))

    # dataset/transform setting
    if args.in_dataset in ['cifar10']:
        args.num_classes=10
    elif args.in_dataset in ['cifar100']:
        args.num_classes=100

    # Get Dataloader
    train_proj_dataloader, train_linear_dataloader, test_dataloader = globals()[args.in_dataset](args,mode = 'train')
   
    # Get architecture
    net = get_architecture(args)

    # Get optimizer, scheduler
    optimizer_proj, scheduler_proj = get_optim_scheduler(args,net)
    
    # Define Loss
    XentLoss = nn.CrossEntropyLoss()
    if args.mode == 'SupCon':
        ContrastiveLoss = SupConLoss(temperature=0.1,mode = args.mode, mixup = args.mixup)
    elif args.mode == 'SimCLR':
        ContrastiveLoss = SupConLoss(temperature=0.5,mode = args.mode, mixup = args.mixup)

    # Projection Training
    path = './checkpoint/'+args.in_dataset+'/'+args.arch+'_'+args.mode+'_proj_'+str(args.proj_noise_rate)+'_trial_'+args.trial
    if not os.path.isfile(path):
        best_loss = 100.
        for epoch in range(args.epoch):
            loss = projection_train(args, net, train_proj_dataloader, optimizer_proj, scheduler_proj, ContrastiveLoss, epoch, forward_part = 'projection')
            if epoch%10 == 0:
                writer.add_scalar('projection train loss',loss,epoch+1)

            scheduler_proj.step()
            if best_loss>loss:
                best_loss = loss
                if not os.path.isdir('checkpoint/'+args.in_dataset):
                    os.makedirs('checkpoint/'+args.in_dataset)
                torch.save(net.state_dict(), path)
    else:
        checkpoint = torch.load('./checkpoint/'+args.in_dataset+'/'+args.arch+'_'+args.mode+'_proj_'+str(args.proj_noise_rate)+'_trial_'+args.trial)
        net.load_state_dict(checkpoint)
        net.train()

    # Freeze Network except linear layer for classification
    for para in net.parameters():
        para.requires_grad = False
    net.linear.weight.requires_grad = True
    net.linear.bias.requires_grad = True
    
    # Linear Setting
    path = './checkpoint/'+args.in_dataset+'/'+args.arch+'_'+args.mode+'_proj_'+str(args.proj_noise_rate)+'_linear_'+str(args.linear_noise_rate)+'_trial_'+args.trial
    args.lr = 5.
    args.epoch = 100
    optimizer_linear, scheduler_linear = get_optim_scheduler(args,net)

    # Linear Training
    best_acc = 0.
    for epoch in range(args.epoch):
        loss = linear_train(args, net, train_linear_dataloader, optimizer_linear, scheduler_linear, XentLoss, epoch, forward_part = 'linear')
        writer.add_scalar('linear train loss',loss,epoch+1)

        acc,loss = test(args, net, test_dataloader, optimizer_linear, scheduler_linear, XentLoss, epoch, forward_part='linear')
        writer.add_scalar('linear test loss',loss,epoch+1)
        writer.add_scalar('linear test accuracy',acc,epoch+1)
        scheduler_linear.step()
        if best_acc<acc:
            best_acc = acc
            if not os.path.isdir('checkpoint/'+args.in_dataset):
                os.makedirs('checkpoint/'+args.in_dataset)
            torch.save(net.state_dict(), path)
    writer.close()

    print('Train End')
    acc,_ = test(args, net, test_dataloader, optimizer_linear, scheduler_linear, XentLoss, 1, forward_part='linear')
    print('Test End')



def projection_train(args, net, train_proj_dataloader, optimizer, scheduler, SupConLoss, epoch, forward_part):
    net.train()
    train_loss = 0
    p_bar = tqdm(range(train_proj_dataloader.__len__()))
    for batch_idx, (images, labels, true_labels) in enumerate(train_proj_dataloader):
        bsz = labels.shape[0]
        images = torch.cat([images[0],images[1]],dim=0)
        images, labels = images.to(args.device), labels.to(args.device)     
        optimizer.zero_grad()
        if args.mixup:
            randidx = torch.randperm(bsz)
            randidx = torch.cat((randidx,randidx+bsz),dim=0)
            images_perm = images[randidx]
            Beta = diri.Dirichlet(torch.tensor([1. for _ in range(2)]))
            lambdas = Beta.sample(randidx.shape).to(args.device)
            images = (images.T*lambdas[:,0] + images_perm.T*lambdas[:,1]).T
        outputs = net(images, mode = forward_part)
        loss = SupConLoss(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epoch,
                    batch=batch_idx + 1,
                    iter=train_proj_dataloader.__len__(),
                    lr = scheduler.optimizer.param_groups[0]['lr'],
                    loss = train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    return train_loss/train_proj_dataloader.__len__()        # average train_loss

def linear_train(args, net, train_linear_dataloader, optimizer, scheduler, XentLoss, epoch, forward_part):
    net.train()
    train_loss = 0
    p_bar = tqdm(range(train_linear_dataloader.__len__()))
    for batch_idx, (images, labels, true_labels) in enumerate(train_linear_dataloader):
        images, labels = images[0].to(args.device), labels.to(args.device)     
        optimizer.zero_grad()
        outputs = net(images,mode = forward_part)
        loss = XentLoss(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epoch,
                    batch=batch_idx + 1,
                    iter=train_linear_dataloader.__len__(),
                    lr = scheduler.optimizer.param_groups[0]['lr'],
                    loss = train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    return train_loss/train_linear_dataloader.__len__()        # average train_loss

def test(args, net, test_dataloader, optimizer, scheduler, Loss, epoch, forward_part):
    net.eval()
    test_loss = 0
    acc = 0
    p_bar = tqdm(range(test_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(args.device), labels.to(args.device)
            if forward_part=='projection':
                images = torch.cat([images,images],dim=0)
            outputs = net(images,mode = forward_part)
            loss = Loss(outputs, labels)
            test_loss += loss.item()
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=test_dataloader.__len__(),
                    lr = scheduler.optimizer.param_groups[0]['lr'],
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            if forward_part=='linear':
                acc+=sum(outputs.argmax(dim=1)==labels)
    p_bar.close()
    acc = acc/test_dataloader.dataset.__len__()
    if forward_part == 'linear':
        print('Accuracy :'+ '%0.4f'%acc )
        return acc, test_loss/test_dataloader.dataset.__len__()
    elif forward_part == 'projection':
        print('Projection Loss : '+'%0.4f'%(test_loss/test_dataloader.dataset.__len__()))
        return test_loss/test_dataloader.dataset.__len__()

if __name__ == '__main__':
    main()