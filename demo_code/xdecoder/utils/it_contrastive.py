import torch
import torch.nn as nn
import torch.nn.functional as F

def is_dist_initialized():
    return torch.distributed.is_initialized()

def get_world_size():
    if is_dist_initialized():
        return torch.distributed.get_world_size()
    return 1

def all_gather_grad(x):
    if get_world_size() > 1:
        all_x = [torch.zeros_like(x) for _ in range(get_world_size())]
        torch.distributed.all_gather(all_x, x)
        all_x[torch.distributed.get_rank()] = x
        x = torch.cat(all_x, dim=0)
    return x

@torch.no_grad()
def all_gather_nograd(tensor):
    # from albef
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if get_world_size() > 1:
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        tensor = torch.cat(tensors_gather, dim=0)
    return tensor

def image_text_contrastive_loss(image_feat, text_feat, temperature, image_id=None, text_id=None):
    # add the following 4 lines
    image_feat = all_gather_grad(image_feat)
    text_feat = all_gather_grad(text_feat)
    
    logits = torch.matmul(image_feat, text_feat.t())
    logits /= temperature
    
    if image_id is None and text_id is None:
        gt = torch.arange(logits.shape[0], device=logits.device)
        loss1 = F.cross_entropy(logits, gt)
        loss2 = F.cross_entropy(logits.t(), gt)        
    else:
        image_id = all_gather_grad(image_id)
        text_id = all_gather_grad(text_id)

        gt_image = image_id.reshape((-1, 1)) == image_id.reshape((1, -1))
        gt_text = text_id.reshape((-1, 1)) == text_id.reshape((1, -1))
        gt = torch.logical_or(gt_image, gt_text)

        loss1 = -torch.sum(gt * F.log_softmax(logits, dim=1)) / gt.sum()
        loss2 = -torch.sum(gt.t() * F.log_softmax(logits.t(), dim=1)) / gt.sum()

    return (loss1 + loss2) / 2 * get_world_size() # scale it up by the number of GPUs
