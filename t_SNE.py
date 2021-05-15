#%%
import torch
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import argparse
from utils.arguments import get_t_SNE_arguments
from utils.utils import *
from dataset.cifar import *
from tqdm import tqdm

def main():
    args = argparse.ArgumentParser()
    args = get_t_SNE_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # dataset/transform setting
    if args.in_dataset in ['cifar10']:
        args.num_classes=10
    elif args.in_dataset in ['cifar100']:
        args.num_classes=100

    # Get Dataloader
    dataloader_list = globals()[args.in_dataset](args)
    test_dataloader = dataloader_list[-1]

    # Get architecture
    net = get_architecture(args)

    # Get optimizer, scheduler
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10,15],gamma=0.2)

    # Get model ckpt
    if args.mode in ['SupCon', 'SimCLR']:
        checkpoint = torch.load('./checkpoint/'+args.in_dataset+'/'+args.arch+'_'+args.mode+'_proj_'+str(args.proj_noise_rate)+'_linear_'+str(args.linear_noise_rate)+'_trial_'+args.trial+args.mixup)
    elif args.mode in ['Xent']:
        checkpoint = torch.load('./checkpoint/'+args.in_dataset+'/'+args.arch+'_'+args.mode+'_'+str(args.noise_rate)+'_trial_'+args.trial+args.mixup)
    net.load_state_dict(checkpoint)
    net.eval()

    # projection drawing
    test_features_proj, y = test(args, net, test_dataloader, optimizer, scheduler, forward_part='projection')
    draw_tSNE(test_features_proj, y,'projection',args)

    # linear drawing
    test_features_linear, y = test(args, net, test_dataloader, optimizer, scheduler, forward_part='linear')
    draw_tSNE(test_features_linear, y,'linear',args)
  

def test(args, net, test_dataloader, optimizer, scheduler, forward_part):
    net.eval()
    p_bar = tqdm(range(test_dataloader.__len__()))
    test_features = torch.Tensor().to(args.device)
    y = torch.Tensor().to(args.device)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images,mode = forward_part)
            test_features = torch.cat((test_features,outputs),dim=0)
            y = torch.cat((y,labels),dim=0)
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. ".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=test_dataloader.__len__(),
                   ))
            p_bar.update()
    p_bar.close()
    print('forward part : '+forward_part)
    return test_features, y

def draw_tSNE(test_features,y,embedding_mode,args):
    test_features = test_features.cpu().numpy()
    y = y.cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
    tsne_ref = tsne.fit_transform(test_features)
    df = pd.DataFrame(tsne_ref, index=tsne_ref[0:,1])
    df['x'] = tsne_ref[:,0]
    df['y'] = tsne_ref[:,1]
    df['Label'] = y[:]
    # sns.scatterplot(x="x", y="y", hue="y", palette=sns.color_palette("hls", 10), data=df)
    if args.mode in ['SimCLR','SupCon']:
        sns.lmplot(x="x", y="y", data=df, fit_reg=False, legend=True, size=9, hue='Label', scatter_kws={"s":200, "alpha":0.5}).savefig('t_SNE/'+args.mode+'_proj_noise_'+str(args.proj_noise_rate)+'_linear_noise_'+str(args.linear_noise_rate)+'_'+embedding_mode+args.mixup+'.png')
    elif args.mode in ['Xent']:
        sns.lmplot(x="x", y="y", data=df, fit_reg=False, legend=True, size=9, hue='Label', scatter_kws={"s":200, "alpha":0.5}).savefig('t_SNE/'+args.mode+'_noise_'+str(args.noise_rate)+'_'+embedding_mode+args.mixup+'.png')
    
    plt.title('t-SNE result', weight='bold').set_fontsize('14')
    plt.xlabel('x', weight='bold').set_fontsize('10')
    plt.ylabel('y', weight='bold').set_fontsize('10')
    plt.show()

if __name__ == '__main__':
    main()
# %%
