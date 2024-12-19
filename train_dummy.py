import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

from torch.utils.data import DataLoader
from xrd_ae.datasets.data import QueueDataset
from xrd_ae.utils.checkpoint import Checkpoint
from xrd_ae.criterion import TotalLoss
from xrd_ae.modeling.ae import ViTAutoencoder

from functools  import partial
from contextlib import nullcontext

import argparse

import traceback

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dummy Data Consumer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--input_queue_name", type=str, default="input",
                        help="Name of the Ray queue to pull raw data from")
    parser.add_argument("--ray_address", type=str, default="auto",
                        help="Address of the Ray cluster")
    parser.add_argument("--ray_namespace", type=str, default="my",
                        help="Ray namespace to use for both queues")
    return parser.parse_args()

args = parse_arguments()

# -- Dataset
dataset = QueueDataset(queue_name=args.input_queue_name, ray_namespace=args.ray_namespace)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
)

# -- Model
model = ViTAutoencoder(
    image_size=(1920, 1920),
    patch_size=128,
    latent_dim=256,
    dim=1024,
    depth=1,
    use_flash=True,
    norm_pix=True,
)
logger.info(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# -- Criterion
kernel_size = 3
weight_factor = 0.5
min_distance = 0.1
div_weight = 0.01
criterion = TotalLoss(kernel_size, weight_factor, min_distance, div_weight)

# -- Optim
def cosine_decay(initial_lr: float, current_step: int, total_steps: int, final_lr: float = 0.0) -> float:
    # Ensure we don't go past total steps
    current_step = min(current_step, total_steps)

    # Calculate cosine decay
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / total_steps))

    # Calculate decayed learning rate
    decayed_lr = final_lr + (initial_lr - final_lr) * cosine_decay

    return decayed_lr

init_lr = 1e-3
weight_decay = 0
adam_beta1 = 0.9
adam_beta2 = 0.999
param_iter = model.parameters()
optim_arg_dict = dict(
    lr           = init_lr,
    weight_decay = weight_decay,
    betas        = (adam_beta1, adam_beta2),
)
optimizer = optim.AdamW(param_iter, **optim_arg_dict)

# -- Device
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
model.to(device)

# -- Misc
# --- Mixed precision
dist_dtype = 'bfloat16'
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dist_dtype]

# --- Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = mixed_precision_dtype)

scaler_func = torch.cuda.amp.GradScaler
scaler = scaler_func(enabled=(dist_dtype == 'float16'))

# --- Grad clip
grad_clip = 1.0

## # --- Normlization
## normalizer = InstanceNorm(scales_variance=True)

# --- Checkpoint
checkpointer = Checkpoint()
path_chkpt = f"chkpt.train_dummy.{os.getenv('CUDA_VISIBLE_DEVICES')}"

# --- Memory
def log_memory():
    if torch.cuda.is_available():
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_cached() / 1e9:.2f} GB")

# -- Training loop
iteration_counter = 0
total_iterations  = 100000
loss_min = float('inf')
while True:
    torch.cuda.synchronize()

    # Adjust learning rate
    lr = cosine_decay(init_lr, iteration_counter, total_iterations*0.5, init_lr*1e-3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for batch_tuple in dataloader:
        rank, id, batch = batch_tuple
        if iteration_counter % 100 == 0:  # Log every few iterations
            log_memory()

        batch = batch.to(device, non_blocking=True)
        ## batch = normalizer(batch)

        logger.info(f"Processing batch {iteration_counter} of shape: {batch.shape}")

        batch[...,:10,:]=0
        batch[...,-10:,:]=0
        batch[...,:,:10]=0
        batch[...,:,-10:]=0

        # Fwd/Bwd
        with autocast_context:
            ## batch_logits = model(batch)
            ## loss = criterion(batch_logits, batch)

            latent = model.encode(batch)
            batch_logits = model.decode(latent)
            loss = criterion(batch, latent, batch_logits)
        scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update parameters
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        # Log
        log_data = {
            "logevent"        : "LOSS:TRAIN",
            "iteration"       : iteration_counter,
            "lr"              : f"{lr:06f}",
            "grad_norm"       : f"{grad_norm:.6f}",
            "mean_train_loss" : f"{loss:.6f}",
        }
        log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
        logger.info(log_msg)

        iteration_counter += 1

        if ((iteration_counter+1)%100 == 0) and (loss_min > loss.item()):
            loss_min = loss.item()
            checkpointer.save(0, model, optimizer, None, None, path_chkpt)
            logger.info(f"--> Saving chkpts to {path_chkpt}")
    if iteration_counter > total_iterations:
        break
