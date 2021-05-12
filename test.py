import torch
import argparse
from dataset.cifar import *
from utils.arguments import get_test_arguments
from utils.utils import *
from dataset.cifar import *
from tqdm import tqdm

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_test_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)

    # dataset setting
    if args.in_dataset in ['cifar10']:
        args.num_classes=10
    elif args.in_dataset in ['cifar100']:
        args.num_classes=100

    # Get Dataloader
    _, test_dataloader = globals()[args.in_dataset](args, mode = 'eval')

    # Get architecture
    net = get_architecture(args)
    path = './checkpoint/'+args.in_dataset+'/'+args.arch+'_'+args.mode+'_proj_'+str(args.proj_noise_rate)+'_linear_'+str(args.linear_noise_rate)+'_trial_'+args.trial
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint)
    net.eval()

    optimizer_linear, scheduler_linear = get_optim_scheduler(args,net)

    # Define Loss
    XentLoss = nn.CrossEntropyLoss()
    test(args, net, test_dataloader, optimizer_linear, scheduler_linear, XentLoss, 1, forward_part='linear')
      

def test(args, net, test_dataloader, optimizer, scheduler, Loss, epoch,forward_part):
    net.eval()
    test_loss = 0
    acc = 0
    p_bar = tqdm(range(test_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images, mode = forward_part)
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
            acc+=sum(outputs.argmax(dim=1)==labels)
    p_bar.close()
    acc = acc/test_dataloader.dataset.__len__()
    if forward_part == 'linear':
        print('Accuracy :'+ '%0.4f'%acc )
        return acc
    elif forward_part == 'projection':
        print('Loss : '+'%0.4f'%(test_loss*256/test_dataloader.dataset.__len__()))
        return test_loss*256/test_dataloader.dataset.__len__()

if __name__ == '__main__':
    main()