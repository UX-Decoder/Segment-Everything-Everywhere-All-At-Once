import os
import time
import torch
import pickle
import subprocess

from mpi4py import MPI
import torch.distributed as dist


def apply_distributed(opt):
    if opt['rank'] == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_address = result.decode('utf-8').split()[0]
        master_port = opt['PORT']
    else:
        master_address = None
        master_port = None

    master_address = MPI.COMM_WORLD.bcast(master_address, root=0)
    master_port = MPI.COMM_WORLD.bcast(master_port, root=0)

    if torch.distributed.is_available() and opt['world_size'] > 1:
        init_method_url = 'tcp://{}:{}'.format(master_address, master_port)
        backend = 'nccl'
        world_size = opt['world_size']
        rank = opt['rank']
        torch.distributed.init_process_group(backend=backend,
                                             init_method=init_method_url,
                                             world_size=world_size,
                                             rank=rank)

def init_distributed(opt):
    opt['CUDA'] = opt.get('CUDA', True) and torch.cuda.is_available()
    if 'OMPI_COMM_WORLD_SIZE' not in os.environ:
        # application was started without MPI
        # default to single node with single process
        opt['env_info'] = 'no MPI'
        opt['world_size'] = 1
        opt['local_size'] = 1
        opt['rank'] = 0
        opt['local_rank'] = 0
        opt['master_address'] = '127.0.0.1'
        opt['master_port'] = '8673'
    else:
        # application was started with MPI
        # get MPI parameters
        opt['world_size'] = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        opt['local_size'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        opt['rank'] = int(os.environ['OMPI_COMM_WORLD_RANK'])
        opt['local_rank'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    # set up device
    if not opt['CUDA']:
        assert opt['world_size'] == 1, 'multi-GPU training without CUDA is not supported since we use NCCL as communication backend'
        opt['device'] = torch.device("cpu")
    else:
        torch.cuda.set_device(opt['local_rank'])
        opt['device'] = torch.device("cuda", opt['local_rank'])

    apply_distributed(opt)
    return opt

def is_main_process():
    rank = 0
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

    return rank == 0

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size == 1:
        return

    def _send_and_wait(r):
        if rank == r:
            tensor = torch.tensor(0, device="cuda")
        else:
            tensor = torch.tensor(1, device="cuda")
        dist.broadcast(tensor, r)
        while tensor.item() == 1:
            time.sleep(1)

    _send_and_wait(0)
    # now sync on the main process
    _send_and_wait(1)