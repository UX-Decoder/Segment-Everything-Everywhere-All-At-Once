import os
import time
import torch
import pickle
import torch.distributed as dist


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


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def broadcast_data(data):
    if not torch.distributed.is_initialized():
        return data
    rank = dist.get_rank()
    if rank == 0:
        data_tensor = torch.tensor(data + [0], device="cuda")
    else:
        data_tensor = torch.tensor(data + [1], device="cuda")
    torch.distributed.broadcast(data_tensor, 0)
    while data_tensor.cpu().numpy()[-1] == 1:
        time.sleep(1)

    return data_tensor.cpu().numpy().tolist()[:-1]


def reduce_sum(tensor):
    if get_world_size() <= 1:
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor