import pickle
from sklearn.cluster import KMeans
import random
import torch.backends.cudnn as cudnn
import os
import glob
import time
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from CloudPatchDataset import CloudPatchDataset 
from torchvision import transforms
import segmentation_models_pytorch as smp
import torch.multiprocessing as mp
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def get_features(dataloader, model, rank, world_size):
    model.eval()
    torch.set_grad_enabled(False)

    local_latent = []
    local_hist = []
    local_meta = []

    for batch in dataloader:
        imgs = batch['image'].to(device)
        hists = batch['hist'].to(device)
        feats = model.encoder(imgs)
        deep_feat = feats[-1]
        pooled = deep_feat.mean(dim=(2, 3))  # (B, D)

        local_latent.append(pooled)
        local_hist.append(hists)
        local_meta.extend(zip(batch['source_file'], batch['row'], batch['col']))

    local_latent = torch.cat(local_latent, dim=0)
    local_hist = torch.cat(local_hist, dim=0)
    local_combined = torch.cat([local_latent, local_hist], dim=1)  # (N_local, D + 3)

    # Gather tensors to rank 0
    gathered_feats = [torch.zeros_like(local_combined) for _ in range(world_size)]
    dist.all_gather(gathered_feats, local_combined)
    combined_feats = torch.cat(gathered_feats, dim=0) if rank == 0 else None

    # Gather metadata separately (not supported by torch.distributed)
    gathered_meta = None
    if rank == 0:
        gathered_meta = [None for _ in range(world_size)]
    gathered_meta = dist.gather_object(local_meta, object_gather_list=gathered_meta if rank == 0 else None, dst=0)

    if rank == 0:
        combined_feats = combined_feats.cpu().numpy()
        combined_feats = StandardScaler().fit_transform(combined_feats)
        all_meta = sum(gathered_meta, [])
        return combined_feats, all_meta
    else:
        return None, None

def load_model(ckpt_loc, use_ckpt, use_recent, rank, cfg, exp_num):

    arch = cfg['architecture']
    encoder = cfg['encoder']
    lr = cfg['lr']

    model = smp.create_model( # create any model architecture just with parameters, without using its class
            arch=arch,
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3, # model input channels
            classes=3 # model output channels
    )
    model = model.to(rank)

    if rank == 0:
        print(summary(model, input_size=(8,3,256,256)))

    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    start_epoch = 0
    best_loss = 0
    ckpt_pth = None

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if use_ckpt:
        if use_recent:
            ckpt_list = glob.glob('{}{}_{}_exp{}_*.pth'.format(ckpt_loc, arch, encoder, exp_num))
            ckpt_list.sort() # sort by time
            if ckpt_list:
                most_recent = ckpt_list.pop()
                ckpt_pth = most_recent
        else:
            ckpt_pth = ckpt_loc
        if ckpt_pth:
            if rank == 0:
                print('using this checkpoint: ', ckpt_pth)
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint=torch.load(ckpt_pth, map_location=map_location, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']

    return model, optimizer, start_epoch, best_loss

def prepare_dataloader(rank, world_size, data_dict, cat, batch_size, pin_memory=True, num_workers=4, is_train=True, train_aug=None):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = CloudDataset(data_dict[cat], transform=data_transforms)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=sampler)
    return dataloader

def main(rank, world_size, config_fn):

    exp_num = config_fn.split('exp')[-1].split('.json')[0]
    with open(config_fn) as fn:
        cfg = json.load(fn)
    arch = cfg['architecture']
    encoder = cfg['encoder']
    lr = cfg['lr']


    data_fn = cfg['datapointer']
    with open(data_fn, 'rb') as handle:
        data_dict = pickle.load(handle)

    lr = cfg['lr']
    batch_size = int(cfg['batch_size'])
    num_workers = int(cfg['num_workers'])
    encoder_weights = cfg['encoder_weights']

    setup(rank, world_size)


    if rank==0:
        print('data dict:              ', data_fn)
        print('config fn:              ', config_fn)
        print('number of train samples:', len(data_dict['train']['truth']))
        print('number of val samples:  ', len(data_dict['val']['truth']))
        print('number of test samples:  ', len(data_dict['test']['truth']))
        print('learning rate:          ', lr)
        print('batch_size:             ', batch_size)
        print('arch:                   ', arch)
        print('encoder:                ', encoder)
        print('num workers:            ', num_workers)
        print('num gpus:               ', world_size)

    #use_ckpt = False
    #use_recent = False
    use_ckpt = True
    use_recent = True
    ckpt_save_loc = './models/'
    ckpt_loc = None
    if use_ckpt:
        if use_recent:
            ckpt_loc = ckpt_save_loc
        else:
            ckpt_loc = cfg['ckpt']

    model, optimizer, start_epoch, best_loss = load_model(ckpt_loc, use_ckpt, use_recent, rank, cfg, exp_num)

    criterion = nn.BCEWithLogitsLoss().to(rank)

    prev_iou = 0

    test_loader = prepare_dataloader(rank, world_size, data_dict, 'test', batch_size=batch_size, is_train=False, num_workers=num_workers)
    scaler = torch.cuda.amp.GradScaler()


    if rank==0:
        start = time.time()

    test_loader.sampler.set_epoch(start_epoch)

    combined_feats, meta_data = get_features(test_loader, model, rank)
    
    finish_feats = time.time()

    if rank == 0:
        print("time to get features:", np.round( finish_feats - start, 2))
        print("Running KMeans clustering...")
        kmeans = KMeans(n_clusters=5, random_state=0)
        cluster_labels = kmeans.fit_predict(combined_feats)

        # save the model and labels
        with open("kmeans_model.pkl", "wb") as f:
            pickle.dump(kmeans, f)
        np.save("cluster_labels.npy", cluster_labels)

    if rank==0:
        print("time to fit kmeans:", np.round( time.time() - finish_feats, 2))

    torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = False
    world_size = 2 # num gpus
    if len(sys.argv) < 2:
        print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
        sys.exit(1)
    config_fn = str(sys.argv[1])
    mp.spawn(main, args=(world_size, config_fn), nprocs=world_size, join=True)

