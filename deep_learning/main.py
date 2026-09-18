import pickle
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
from CloudDataset import CloudDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#CTT
#TASKS = { 'cod': (0, 5), 'ctt': (5, 10), 'ctp': (10, 15), 'transition': (15, 19) }
TASKS = { 'cod': (0, 5), 'ctt': (5, 10), 'ctp': (10, 15)}


def thermometer_target(truth):
    levels = torch.arange(1, 6, device=truth.device).view(1, 5, 1, 1)
    return (truth.unsqueeze(1) >= levels).float()


def ordinal_loss(pred, truth, criterion):
    valid = truth > 0
    target = thermometer_target(truth)
    loss = criterion(pred, target)
    return loss[valid.unsqueeze(1).expand_as(loss)].mean()


class IoUCalculator:
    def __init__(self, num_classes, ordinal=False):
        self.num_classes = num_classes
        self.ordinal = ordinal
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self.union = torch.zeros(self.num_classes, dtype=torch.float64)

    def update(self, pred, truth):
        if self.ordinal:
            pred = (torch.sigmoid(pred) > 0.5).sum(dim=1)
            valid = truth > 0
            pred = pred[valid]
            truth = truth[valid]
            classes = range(1, self.num_classes + 1)
        else:
            pred = torch.argmax(pred, dim=1)
            classes = range(self.num_classes)

        for c in classes:
            pred_c = pred == c
            truth_c = truth == c
            self.intersection[c - 1 if self.ordinal else c] += (pred_c & truth_c).sum().item()
            self.union[c - 1 if self.ordinal else c] += (pred_c | truth_c).sum().item()

    def all_reduce(self, rank):
        device = torch.device(f"cuda:{rank}")
        intersection = self.intersection.to(device)
        union = self.union.to(device)
        dist.all_reduce(intersection, dist.ReduceOp.SUM)
        dist.all_reduce(union, dist.ReduceOp.SUM)
        return intersection.cpu(), union.cpu()


def print_iou(name, intersection, union, ordinal=False):
    ious = []
    for i in range(len(intersection)):
        iou = float('nan') if union[i] == 0 else intersection[i].item() / union[i].item()
        ious.append(iou)
        label = i + 1 if ordinal else i
        print(f"{name} class {label} IoU: {'nan' if np.isnan(iou) else f'{iou:.4f}'}")
    valid = [x for x in ious if not np.isnan(x)]
    mean_iou = np.mean(valid) if valid else float('nan')
    print(f"{name} mean IoU: {'nan' if np.isnan(mean_iou) else f'{mean_iou:.4f}'}")
    return mean_iou


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def calculate_task_losses(preds, labels, criterion_ord, criterion_cat):
    losses = {}
    losses['cod'] = criterion_ord(preds[:, 0:5], labels[:, 0:5])
    losses['ctt'] = criterion_ord(preds[:, 5:10], labels[:, 5:10])
    losses['ctp'] = criterion_ord(preds[:, 10:15], labels[:, 10:15])
#    losses['transition'] = criterion_cat(preds[:, 15:19], labels[:, 15].long())
    return losses, sum(losses.values())



def val_model(dataloader, model, criterion_ord, criterion_cat, rank, world_size):
    model.eval()
    total_loss = 0.0
#CTT
    #iou_calculators = { 'cod': IoUCalculator(5, ordinal=True), 'ctt': IoUCalculator(5, ordinal=True), 'ctp': IoUCalculator(5, ordinal=True), 'transition': IoUCalculator(4) }
    iou_calculators = { 'cod': IoUCalculator(5, ordinal=True), 'ctt': IoUCalculator(5, ordinal=True), 'ctp': IoUCalculator(5, ordinal=True)}

    with torch.inference_mode():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(rank, dtype=torch.float32, non_blocking=True)
            batch_labels = batch_labels.to(rank, dtype=torch.float32, non_blocking=True)

            with torch.amp.autocast('cuda'):
                preds = model(batch_data)
                losses, loss = calculate_task_losses(preds, batch_labels, criterion_ord, criterion_cat)

            total_loss += loss.item()

#CTT
            for task in ['cod', 'ctt', 'ctp']:
                start, end = TASKS[task]
                task_idx = ['cod', 'ctt', 'ctp'].index(task)
                iou_calculators[task].update(preds[:, start:end, :, :], batch_labels[:, task_idx, :, :])

#CTT
#            start, end = TASKS['transition']
#            iou_calculators['transition'].update(preds[:, start:end, :, :], batch_labels[:, 3, :, :])

    final_loss = total_loss / len(dataloader)
    loss_tensor = torch.tensor([final_loss], device=rank)
    dist.all_reduce(loss_tensor)
    loss_tensor /= world_size

    results = {}
    for task, calculator in iou_calculators.items():
        results[task] = calculator.all_reduce(rank)

    if rank == 0:
        print(f"Validation Loss: {loss_tensor[0].item():.4f}", flush=True)
        task_ious = []
        for task, (intersection, union) in results.items():
            mean_iou = print_iou(
                task.upper(),
                intersection,
                union,
                ordinal=(task != 'transition')
            )
            if not np.isnan(mean_iou):
                task_ious.append(mean_iou)
        overall_iou = np.mean(task_ious) if task_ious else 0
        print(f"Overall mean IoU: {overall_iou:.4f}", flush=True)

    return final_loss, results


def load_model(ckpt_loc, use_ckpt, use_recent, rank, cfg, exp_num):
    arch = cfg['architecture']
    encoder = cfg['encoder']
    lr = cfg['lr']

    model = smp.create_model(arch=arch, encoder_name=encoder, encoder_weights=None, in_channels=7, classes=5)
#CTT
    #model = smp.create_model(arch=arch, encoder_name=encoder, encoder_weights=None, in_channels=7, classes=19)
    model = model.to(rank)

#    if rank == 0:
#        print(summary(model, input_size=(8, 7, 256, 256)))

    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    start_epoch = 0
    best_loss = 0
    ckpt_pth = None

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if use_ckpt:
        if use_recent:
            ckpt_list = glob.glob(f'{ckpt_loc}{arch}_{encoder}_exp{exp_num}_*.pth')
            ckpt_list.sort()
            if ckpt_list:
                ckpt_pth = ckpt_list.pop()
        else:
            ckpt_pth = ckpt_loc

        if ckpt_pth:
            if rank == 0:
                print('using this checkpoint:', ckpt_pth)
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(ckpt_pth, map_location=map_location, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']

    return model, optimizer, start_epoch, best_loss


def train_model(train_dataloader, model, criterion_ord, criterion_cat, optimizer, rank, scaler):
    total_loss = 0.0
    model.train()
    torch.set_grad_enabled(True)
    start = time.time()

    for batch_data, batch_labels in train_dataloader:
        optimizer.zero_grad()
        batch_data = batch_data.to(rank, dtype=torch.float32, non_blocking=True)
        batch_labels = batch_labels.to(rank, dtype=torch.float32, non_blocking=True)

        with torch.amp.autocast('cuda'):
            preds = model(batch_data)
            losses, loss = calculate_task_losses(preds, batch_labels, criterion_ord, criterion_cat)

        total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    epoch_loss = total_loss / len(train_dataloader)

    if rank == 0:
        print('training time:', np.round(time.time() - start, 2), flush=True)
        print('training loss:', np.round(epoch_loss, 4), flush=True)


def get_subset_train(data_dict):
    subset_data_dict = {'train': {'data': [], 'truth': []}}
    num_samples = int(len(data_dict['train']['truth']) / 5)
    subset_data_dict['train']['data'] = random.sample(data_dict['train']['data'], num_samples)
    subset_data_dict['train']['truth'] = random.sample(data_dict['train']['truth'], num_samples)
    return subset_data_dict


def get_transforms(train_augs):
    transform_list = [transforms.ToTensor()]
    if 'rhf' in train_augs.keys():
        transform_list.append(transforms.RandomHorizontalFlip(p=train_augs['rhf']))
    if 'rvf' in train_augs.keys():
        transform_list.append(transforms.RandomVerticalFlip(p=train_augs['rvf']))
    return transforms.Compose(transform_list)


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

    n_epochs = 100
    start_epoch = 0
    batch_size = int(cfg['batch_size'])
    num_workers = int(cfg['num_workers'])
    encoder_weights = cfg['encoder_weights']

    setup(rank, world_size)

    if rank == 0:
        print('data dict:              ', data_fn)
        print('config fn:              ', config_fn)
        print('number of train samples:', len(data_dict['train']['truth']))
        print('number of val samples:  ', len(data_dict['val']['truth']))
        print('number of test samples: ', len(data_dict['test']['truth']))
        print('learning rate:          ', lr)
        print('batch_size:             ', batch_size)
        print('arch:                   ', arch)
        print('encoder:                ', encoder)
        print('num workers:            ', num_workers)
        print('num gpus:               ', world_size)

    use_ckpt = False
    use_recent = False
    ckpt_save_loc = './models/'
    ckpt_loc = None

    if use_ckpt:
        if use_recent:
            ckpt_loc = ckpt_save_loc
        else:
            ckpt_loc = cfg['ckpt']

    model, optimizer, start_epoch, best_loss = load_model(
        ckpt_loc, use_ckpt, use_recent, rank, cfg, exp_num
    )

    criterion_ord = nn.BCEWithLogitsLoss().to(rank)
    criterion_cat = nn.CrossEntropyLoss().to(rank)

    train_loader = prepare_dataloader(
        rank, world_size, data_dict, 'train',
        batch_size=batch_size,
        num_workers=num_workers,
        train_aug=cfg['train_augmentations']
    )

    val_loader = prepare_dataloader(
        rank, world_size, data_dict, 'val',
        batch_size=batch_size,
        is_train=False,
        num_workers=num_workers
    )

    scaler = torch.cuda.amp.GradScaler()
    prev_iou = 0

    for epoch in range(start_epoch, n_epochs):
        if rank == 0:
            print(f'--------------\nStarting Epoch: {epoch}', flush=True)
            start = time.time()

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        train_model(train_loader, model, criterion_ord, criterion_cat, optimizer, rank, scaler)
        val_loss, iou_results = val_model(val_loader, model, criterion_ord, criterion_cat, rank, world_size)

        if rank == 0:
            print("time to run epoch:", np.round(time.time() - start, 2))

            task_ious = []
            for task, (intersection, union) in iou_results.items():
                ious = [intersection[i].item() / union[i].item() for i in range(len(intersection)) if union[i] != 0]
                if ious:
                    task_ious.append(np.mean(ious))

            iou = np.mean(task_ious) if task_ious else 0
            print("overall IoU:", iou)

            if iou > prev_iou and iou > .40:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'iou': iou
                }

                ckpt_pth = f'{ckpt_save_loc}{arch}_{encoder}_exp{exp_num}_{int(time.time())}.pth'
                torch.save(checkpoint, ckpt_pth)
                print('SAVING MODEL:\n', ckpt_pth, flush=True)
                prev_iou = iou

        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = False
    world_size = 2

    if len(sys.argv) < 2:
        print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
        sys.exit(1)

    config_fn = str(sys.argv[1])
    mp.spawn(main, args=(world_size, config_fn), nprocs=world_size, join=True)
