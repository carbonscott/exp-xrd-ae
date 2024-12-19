import torch
import os
import time
import ray
from torch.utils.data import DataLoader
from xrd_ae.datasets.data import QueueDataset
from mpi4py import MPI

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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--input_queue_name", type=str, default="input",
                        help="Name of the Ray queue to pull raw data from")
    parser.add_argument("--ray_address", type=str, default="auto",
                        help="Address of the Ray cluster")
    parser.add_argument("--ray_namespace", type=str, default="my",
                        help="Ray namespace to use for both queues")
    return parser.parse_args()

args = parse_arguments()

dataset = QueueDataset(queue_name=args.input_queue_name, ray_namespace=args.ray_namespace)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
)

batch_idx = 0
for batch in dataloader:
    rank, id, data = batch
    logger.info(f"Processing batch {batch_idx} of shape: {data.shape}")
    batch_idx += 1
