import pickle
from distutils import log

import torch
import torch.nn.functional as F
import torch.distributed as dist

from einops import rearrange, repeat
from timm.loss import SoftTargetCrossEntropy

soft_cross_entropy = SoftTargetCrossEntropy()

def is_dist_initialized():
    return torch.distributed.is_initialized()

def get_world_size():
    if is_dist_initialized():
        return torch.distributed.get_world_size()
    return 1

def get_rank():
    if is_dist_initialized():
        return dist.get_rank()
    return 0

def all_gather_grad(x):
    if get_world_size() > 1:
        all_x = [torch.zeros_like(x) for _ in range(get_world_size())]
        torch.distributed.all_gather(all_x, x)
        all_x[torch.distributed.get_rank()] = x
        x = torch.cat(all_x, dim=0)
    return x

def vl_multilabel_contrastive_loss(image_feat, text_feat, temperature=1):
    """
    Args:
        image_feat (torch.Tensor): shape [B, L1, C] # B: batch_size, L1: 1, C: 256
        text_feat (torch.Tensor): shape [B, L2, C] # B:batch_size, L2: number of selected nouns, C: 256

    Returns:
    """
    # [B, L1, C], L1 = 1
    # image_feat = F.normalize(image_feat, dim=-1)
    # [B, L2, C]
    # text_feat = F.normalize(text_feat, dim=-1)
    # HACK: normalize outside
    
    # [B, L1, L2]
    dist_per_img = image_feat @ rearrange(text_feat, 'b l c -> b c l')    
    # [B, L2, L1]
    dist_per_text = text_feat @ rearrange(image_feat, 'b l c -> b c l')

    batch = image_feat.shape[0]
    img_len = image_feat.shape[1]
    text_len = text_feat.shape[1]
    # [B, L1, L2]
    pos_labels_batch_img = rearrange(torch.ones_like(dist_per_text) / dist_per_text.size(1), 'b l2 l1 -> b l1 l2')
    # [B, L2, L1]
    pos_labels_batch_text = rearrange(torch.ones_like(dist_per_img) / dist_per_img.size(1), 'b l1 l2 -> b l2 l1')

    image_x = rearrange(image_feat, 'b l c -> (b l) c')
    text_x = rearrange(text_feat, 'b l c -> (b l) c')

    logits_per_img = image_x @ all_gather_grad(text_x).t()
    logits_per_text = text_x @ all_gather_grad(image_x).t()

    # get label globally
    # [B, L1, B, L2, W]
    labels_per_img = F.one_hot(
        torch.ones(batch, img_len, batch, text_len, dtype=torch.long, device=image_x.device) * get_rank(),
        num_classes=get_world_size()).to(image_x.dtype)
    labels_per_img *= rearrange(pos_labels_batch_img, 'b l1 l2 -> b l1 1 l2 1') * repeat(
        torch.eye(batch, dtype=image_x.dtype, device=image_x.device), 'b1 b2 -> b1 1 b2 1 1')
    # [BxL1, WxBxL2]
    labels_per_img = rearrange(labels_per_img, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
    # [B, L2, B, L1, W]
    labels_per_text = F.one_hot(
        torch.ones(batch, text_len, batch, img_len, dtype=torch.long, device=text_x.device) * get_rank(),
        num_classes=get_world_size()).to(text_x.dtype)
    labels_per_text *= rearrange(pos_labels_batch_text, 'b l2 l1 -> b l2 1 l1 1') * repeat(
        torch.eye(batch, dtype=text_x.dtype, device=image_x.device), 'b2 b1 -> b2 1 b1 1 1')
    # [BxL2, WxBxL1]
    labels_per_text = rearrange(labels_per_text, 'b2 l2 b1 l1 w -> (b2 l2) (w b1 l1)')

    logit_scale = temperature.exp().clamp(max=100)

    loss_img = soft_cross_entropy(logit_scale * logits_per_img, labels_per_img)
    loss_text = soft_cross_entropy(logit_scale * logits_per_text, labels_per_text)

    loss = 0.5 * (loss_img + loss_text)
    return loss

def vl_contrastive_loss(image_feat, text_feat, temperature=1):
    # if image_id or text_id is None, it should be None across all GPUs
    # image_feat = F.normalize(image_feat, dim=1)
    # text_feat = F.normalize(text_feat, dim=1)
    # handle normalization outside

    # add the following 4 lines
    image_feat = all_gather_grad(image_feat)
    text_feat = all_gather_grad(text_feat)
    
    logits = torch.matmul(image_feat, text_feat.t())
    logit_scale = temperature.exp().clamp(max=100)

    gt = torch.arange(logits.shape[0], device=logits.device)
    loss1 = F.cross_entropy(logit_scale * logits, gt)
    loss2 = F.cross_entropy(logit_scale * logits.t(), gt)
    return (loss1 + loss2) / 2 # scale it up by the number of GPUs


def all_gather_pickle(data, device):
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
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).cuda()
    size_list = [torch.LongTensor([0]).cuda() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).cuda())
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def all_gather_arbitary_tensor(tensor):
    if get_world_size() > 1:
        device = tensor.device
        tensor_batch = all_gather_pickle(tensor.cpu(), device)
        tensor_batch = [x.to(device) for x in tensor_batch]
        tensor_batch[torch.distributed.get_rank()] = tensor
        tensor_batch = torch.cat(tensor_batch, dim=0)
    else:
        tensor_batch = tensor
    return tensor_batch

def ql_contrastive_loss(image_feat, text_feat, temperature=1):
    # add the following 4 lines
    image_feat = all_gather_arbitary_tensor(image_feat)
    text_feat = all_gather_arbitary_tensor(text_feat)

    logits = torch.matmul(image_feat, text_feat.t())
    logit_scale = temperature.exp().clamp(max=100)

    gt = torch.arange(logits.shape[0], device=logits.device)
    loss1 = F.cross_entropy(logit_scale * logits, gt)
    loss2 = F.cross_entropy(logit_scale * logits.t(), gt)
    return (loss1 + loss2) / 2 # scale it up by the number of GPUs

def vl_similarity(image_feat, text_feat, temperature=1):
    # Only support single GPU for now.
    logits = torch.matmul(image_feat, text_feat.t())
    logits = temperature.exp().clamp(max=100) * logits
    return logits

def ql_multi_contrastive_loss(image_feat, text_feat, text_hash, temperature=1):
    # add the following 4 lines
    image_feat = all_gather_arbitary_tensor(image_feat)
    text_feat = all_gather_arbitary_tensor(text_feat)

    text_hash_batch = all_gather_pickle(text_hash, text_feat.device)
    text_hash_all = torch.cat(text_hash_batch)
    
    text_hash_all_unique = torch.unique(text_hash_all).tolist()
    gt = torch.zeros((image_feat.shape[0], len(text_hash_all_unique)), device=text_feat.device)
    text_hash_all = text_hash_all.tolist()
    text_feat_unique = torch.stack([text_feat[text_hash_all.index(txt)] for txt in text_hash_all_unique])

    for idx, txt in enumerate(text_hash_all):
        gt[idx][text_hash_all_unique.index(txt)] = 1
    
    logits = torch.matmul(image_feat, text_feat_unique.t())
    logits = logits*temperature.exp().clamp(max=100)
    
    loss_img = soft_cross_entropy(logits, gt)
    loss_text = soft_cross_entropy(logits.t(), gt.t() / gt.t().sum(-1, keepdim=True))

    loss = 0.7 * loss_img + 0.3 * loss_text
    return loss

def image_text_contrastive_loss_queue(image_feat_inp, text_feat_inp, lang_enc, training):
    # add the following 4 lines
    image_feat = all_gather_grad(image_feat_inp.contiguous())
    text_feat = all_gather_grad(text_feat_inp.contiguous())

    image_feat = image_feat / (image_feat.norm(dim=-1, keepdim=True) + 1e-7)
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = lang_enc.logit_scale
    logits = torch.matmul(image_feat, text_feat.t())
    logit_scale = temperature.exp().clamp(max=100)

    gt = torch.arange(logits.shape[0], device=logits.device)
    loss1 = F.cross_entropy(logit_scale * logits, gt)
    loss2 = F.cross_entropy(logit_scale * logits.t(), gt)

    return (loss1 + loss2) / 2 # scale it up by the number of GPUs