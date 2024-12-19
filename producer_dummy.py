import ray
from xrd_ae.datasets.queue import Queue

import torch

import argparse
import random
import time
import signal
import sys
import numpy as np
import logging

from mpi4py import MPI

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dummy Data Producer")
    parser.add_argument("--ray_address", type=str, default="auto", help="Address of the Ray cluster")
    parser.add_argument("--ray_namespace", type=str, default="default", help="Ray namespace to use for both queues")
    parser.add_argument("--queue_name", type=str, default='my', help="Queue name")
    parser.add_argument("--queue_size", type=int, default=100, help="Maximum queue size")
    parser.add_argument("--num_consumers", type=int, default=1, help="Number of consumer processes expected.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps before terminating")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    return parser.parse_args()

def initialize_ray(ray_address, ray_namespace, queue_name, queue_size, rank, max_retries=10, retry_delay=1):
    comm = MPI.COMM_WORLD

    try:
        ray.init(address=ray_address, namespace=ray_namespace)
        logging.info(f"Rank {rank}: Ray initialized successfully.")

        if rank == 0:
            try:
                queue = ray.get_actor(queue_name, namespace=ray_namespace)
                logging.info(f"Connected to existing input queue: {queue_name} in namespace: {ray_namespace}")
            except ValueError:  # Ray raises ValueError when actor is not found
                queue = Queue.options(name=queue_name, namespace=ray_namespace).remote(maxsize=queue_size)
                logging.info(f"Rank {rank}: Shared input queue created successfully.")
        else:
            logging.info(f"Rank {rank}: Waiting for shared input queue to be created...")

        # Wait for rank 0 to create the queue
        comm.Barrier()

        # All ranks try to get the queue
        retries = 0
        while retries < max_retries:
            try:
                queue = ray.get_actor(queue_name, namespace=ray_namespace)
                logging.info(f"Rank {rank}: Successfully connected to shared queue.")
                return queue
            except ValueError:  # Actor not found
                retries += 1
                logging.info(f"Rank {rank}: Attempt {retries}/{max_retries} to connect to shared queue failed. Retrying...")
                time.sleep(retry_delay)

        raise TimeoutError(f"Rank {rank}: Timeout waiting for shared_queue actor")

    except Exception as e:
        logging.error(f"Rank {rank}: Error in initialize_ray: {e}")
        return None

def signal_handler(sig, frame):
    logging.error("Ctrl+C pressed. Shutting down...")
    ray.shutdown()
    exit(0)

def produce_data(queue, rank, size, num_consumers=1, max_steps=None):
    comm = MPI.COMM_WORLD

    TOTAL_ITERATIONS = int(1e5)
    C, H, W = 1, 1920, 1920

    # Delay for exponential backoff
    base_delay_in_sec = 0.1
    max_delay_in_sec  = 5.0

    for idx in range(TOTAL_ITERATIONS):
        if max_steps is not None and idx >= max_steps:
            logging.info(f"Rank {rank}: Reached max_steps {max_steps}, terminating")
            break

        # Generate data
        data = torch.randn(C, H, W)  # Simulate a Rayonix image

        retries = 0
        while True:
            try:
                success = ray.get(queue.put.remote([rank, idx, data]))
                if success:
                    logging.info(f"Rank {rank} produced: idx={idx} | shape={data.shape}")
                    break  # Break the while loop and move to the next event
                else:
                    logging.info(f"Rank {rank}: Queue is full, waiting...")
                    # Use exponential backoff with jitter
                    delay_in_sec = min(max_delay_in_sec, base_delay_in_sec * (2 ** retries))
                    jitter_in_sec = random.uniform(0, 0.5)
                    time.sleep(delay_in_sec + jitter_in_sec)
                    if delay_in_sec < max_delay_in_sec: retries += 1
            except ray.exceptions.RayActorError:
                logging.error(f"Rank {rank}: Queue actor is dead. Exiting...")
                return  # Exit the function if the queue actor is dead
            except Exception as e:
                logging.error(f"Rank {rank}: Error in produce_data: {e}")
                time.sleep(1)  # Wait before retrying

    # Signal completion
    comm.Barrier()
    if rank == 0:
        # Put sentinel values to signal end of data (one per consumer)
        try:
            for _ in range(num_consumers):
                ray.get(queue.put.remote(None))
            logging.info(f"Rank 0: {num_consumers} sentinel values sent successfully")
        except ray.exceptions.RayActorError:
            logging.info("Rank 0: Queue actor is dead. Unable to send sentinels.")
        except Exception as e:
            logging.info(f"Rank 0: Error putting sentinel values: {e}")

def main():
    args = parse_arguments()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s - %(levelname)s - %(message)s')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        signal.signal(signal.SIGINT, signal_handler)

    queue = initialize_ray(args.ray_address, args.ray_namespace, args.queue_name, args.queue_size, rank)
    if queue is None:
        MPI.Finalize()
        return

    try:
        produce_data(queue, rank, size, args.num_consumers, args.max_steps)
    except Exception as e:
        logging.error(f"Rank {rank}: Unhandled exception in main: {e}")
    finally:
        if rank == 0:
            ray.shutdown()
        MPI.Finalize()

if __name__ == "__main__":
    main()
