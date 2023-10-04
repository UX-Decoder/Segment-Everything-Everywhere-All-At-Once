import torch
import torch.nn as nn
import torch.nn.functional as F

predict_name_matcher = {"predictions_class": ["pred_logits"],
                        "predictions_mask":["pred_masks", "pred_gmasks", "pred_smasks"],
                        "predictions_caption":["pred_captions", "pred_gtexts", "pred_stexts"],
                        "predictions_maskemb":["pred_smaskembs"],
                        "predictions_pos_spatial":["pred_pspatials"],
                        "predictions_neg_spatial":["pred_nspatials"],}

predict_index_matcher = {"predictions_class": ["queries_object"],
                         "predictions_mask":["queries_object", "queries_grounding", "queries_spatial"],
                         "predictions_caption": ["queries_object", "queries_grounding", "queries_spatial"],
                         "predictions_maskemb":["queries_spatial"],
                         "predictions_pos_spatial":["all"],
                         "predictions_neg_spatial":["all"],}

class Variable(object):
    '''
    Store dataset variable for attention
    output: embedding that accumuates during cross/self attention
    pos: positional embedding that is fixed during cross/self attention
    name: name of the variable
    type: type of the variable, e.g. queries, tokens
    attn_mask: attention mask for corss attention
    masking: masking for padding
    '''
    def __init__(self, output, name, _type, pos=None):
        self.output = output
        self.pos = pos
        self.name = name
        self.type = _type
        self.attn_mask = None
        self.masking = None
    
    def copy(self,):
        output = self.output.clone() if self.output is not None else None
        pos = self.pos.clone() if self.pos is not None else None
        return Variable(output, self.name, self.type, pos)

    def rand_sample(self, max_len):
        rand_idx = torch.randint(0, len(self.pos), (max_len,))
        self.output = self.output[rand_idx]
        self.pos = self.pos[rand_idx]
        return self

class AttentionDataStruct(nn.Module):
    '''
    Store dataset structure for cross/self attention
    task_switch: switch for different tasks

    p_attn_variables: prototype of variables that is used in cross/self attention
    p_self_attn: prototype of variables that is used in self attention
    p_cross_attn: prototype of variables that is used in cross attention
    p_iter: prototype of iteration for different queries
    p_masking: prototype of masking for different tokens
    p_duplication: prototype of duplication for different quries
    '''
    def __init__(self, attn_arch, task_switch):
        super(AttentionDataStruct, self).__init__()
        self.task_switch = task_switch

        # p stands for prototype
        self.p_attn_variables = attn_arch['VARIABLE']
        self.p_self_attn = attn_arch['SELF_ATTENTION']
        self.p_cross_attn = attn_arch['CROSS_ATTENTION']
        self.p_masking = attn_arch['MASKING']
        self.p_duplication = attn_arch['DUPLICATION']

        self.num_layers = attn_arch['NUM_LAYERS']

    def reset(self, flags, task, extra):
        # reset variables
        self.attn_variables = {}
        self.cross_attn_dict = {}
        self.self_attn_dict = {}
        self.duplication_dict = {}
        self.query_index = {}
        self.output = {}
        self.flags = {}
        self.spatial_memory = {}
        self.extra = {}

        # initialize duplication
        for key, values in self.p_duplication.items():
            for name in values:
                self.duplication_dict["{}_{}".format(key, name)] = self.p_duplication[key][name]

        # initialize flag
        self.flags = {"object": True}
        self.flags.update(flags)

        # initialize task
        self.task = task

        # initialize output
        if self.task_switch['mask']:
            self.output['predictions_class'] = []
            self.output['predictions_mask'] = []
        
        if self.task_switch['bbox']:
            self.output['predictions_bbox'] = []

        if self.task_switch['spatial'] and ('memories_spatial' in self.flags and self.flags['memories_spatial']==True):
            self.spatial_memory['prev_batch_mask'] = extra['prev_mask']

        if self.task_switch['grounding'] and ('grounding' in self.flags and self.flags['grounding']==True):
            self.output['predictions_caption'] = []
        
        if self.task_switch['spatial'] and ('spatial' in self.flags and self.flags['spatial']==True):
            self.output['predictions_maskemb'] = []
            self.output['predictions_pos_spatial'] = []
            self.output['predictions_neg_spatial'] = []
            self.output['predictions_mask'] = [] if 'predictions_mask' not in self.output else self.output['predictions_mask']
            self.output['predictions_class'] = [] if 'predictions_class' not in self.output else self.output['predictions_class']
            self.output['predictions_caption'] = [] if 'predictions_caption' not in self.output else self.output['predictions_caption']
        
        # initialize cross_attn, whether the variable is used in cross attention
        for key, values in self.p_cross_attn.items():
            for name in values:
                self.cross_attn_dict["{}_{}".format(key, name)] = self.p_cross_attn[key][name]
        
        # initialize self_attn, whether the variable is used in self attention, and the interactions between queries
        for key, values in self.p_self_attn.items():
            for name in values:
                self.self_attn_dict["{}_{}".format(key, name)] = self.p_self_attn[key][name]
        
        # initialize masking
        self.masking = self.p_masking

        # initialize query_index
        self.query_index = {"all":[0, None]}


    def set(self, name, _type, output=None, pos=None, var=None, sample_size=None):
        if var is not None:
            self.attn_variables[name] = var
        elif name in self.duplication_dict:
            assert self.duplication_dict[name] in self.attn_variables, "Duplication variable {} is not initialized yet.".format(name)
            var = self.attn_variables[self.duplication_dict[name]].copy()
            if sample_size is not None:
                var = var.rand_sample(sample_size)
            self.attn_variables[name] = var
        else:
            var = Variable(output, name, _type, pos)
            self.attn_variables[name] = var
    
    def set_results(self, results):
        for name in self.cross_attn_name:
            self.attn_variables[name].attn_mask = results['attn_mask'][:,self.query_index[name][0]:self.query_index[name][1]]
        for key in self.output:
            self.output[key].append(results[key])
    
    def set_maskings(self, name, masking):
        self.attn_variables[name].masking = masking
    
    def set_extra(self, extra):
        self.extra.update(extra)

    def cross_attn_variables(self, ):
        cross_attn_name = [key for key, value in self.cross_attn_dict.items() 
                           if (value==True) and (key in self.attn_variables) 
                           and ((key not in self.flags) or (key in self.flags and self.flags[key]==True))]
        self.cross_attn_name = cross_attn_name

        output = torch.cat([self.attn_variables[name].output for name in cross_attn_name])
        pos_emb = torch.cat([self.attn_variables[name].pos for name in cross_attn_name])
        
        index = 0
        for name in cross_attn_name:
            self.query_index[name] = [index, index + self.attn_variables[name].output.shape[0]]
            index += self.attn_variables[name].output.shape[0]
        return output, pos_emb
    
    def cross_attn_mask(self, size, num_heads):
        attn_mask = torch.cat([self.attn_variables[name].attn_mask for name in self.cross_attn_name], dim=1)

        # hard code memories_spatial to previous selected mask
        if 'memories_spatial' in self.cross_attn_name:
            memory_attn_mask = self.spatial_memory['prev_batch_mask']
            bs,c,_,_ = memory_attn_mask.shape
            memory_attn_mask = F.interpolate(memory_attn_mask, size, mode='bilinear', align_corners=False)
            memory_attn_mask = (memory_attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, num_heads, 1, 1).flatten(0, 1) < 0.5).bool().detach()
            repeat = (self.query_index['memories_spatial'][1] - self.query_index['memories_spatial'][0]) // c
            mem_len = self.query_index['memories_spatial'][1] - self.query_index['memories_spatial'][0]
            probs = torch.tensor([1./repeat for i in range(c)])
            indices = torch.multinomial(probs, num_samples=mem_len, replacement=True).sort()[0]
            attn_mask[:,self.query_index['memories_spatial'][0]:self.query_index['memories_spatial'][1]] = memory_attn_mask[:,indices]
            self.extra['memory_indices'] = indices

        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        return attn_mask

    def self_attn(self, bs, num_heads):
        self_attn_name = [key for key, value in self.self_attn_dict.items() 
                          if len(value)>0 and key in self.attn_variables
                          and ((key not in self.flags) or (key in self.flags and self.flags[key]==True))]
        self.self_attn_name = self_attn_name

        output = torch.cat([self.attn_variables[name].output for name in self_attn_name])
        pos_emb = torch.cat([self.attn_variables[name].pos for name in self_attn_name])

        index = 0
        for name in self_attn_name:
            self.query_index[name] = [index, index + self.attn_variables[name].output.shape[0]]
            index += self.attn_variables[name].output.shape[0]
        
        self_attn_mask = torch.ones((bs, output.shape[0], output.shape[0]), dtype=torch.bool, device=output.device)
        self_attn_pair = []
        # build self_attention mask by query interaction
        for key1, value in self.self_attn_dict.items():
            for key2 in value:
                if key1 not in self_attn_name or key2 not in self_attn_name:
                    # exclude the variables that are not used in the current layer
                    continue
                if (key1 in self.masking or key2 in self.masking) and (key1 != key2):
                    self_attn_pair += [[key1, key2]]
                self_attn_mask[:,self.query_index[key1][0]:self.query_index[key1][1], self.query_index[key2][0]:self.query_index[key2][1]] = False

        # build self_attention mask by masking, for birectional
        for key in self.masking:
            if key in self_attn_name:
                self_attn_mask[:,self.query_index[key][0]:self.query_index[key][1],self.query_index[key][0]:self.query_index[key][1]][self.attn_variables[key].masking] = True
                self_attn_mask[:,self.query_index[key][0]:self.query_index[key][1],self.query_index[key][0]:self.query_index[key][1]].transpose(1,2)[self.attn_variables[key].masking] = True

        # build self_attention mask by masking, for uni-directional
        for key1, key2 in self_attn_pair:
            if key1 not in self_attn_name or key2 not in self_attn_name:
                # exclude the variables that are not used in the current layer
                continue
            if key1 in self.masking:
                self_attn_mask[:,self.query_index[key1][0]:self.query_index[key1][1],self.query_index[key2][0]:self.query_index[key2][1]][self.attn_variables[key1].masking] = True # HACK, not verified
            if key2 in self.masking:
                self_attn_mask[:,self.query_index[key1][0]:self.query_index[key1][1],self.query_index[key2][0]:self.query_index[key2][1]].transpose(1,2)[self.attn_variables[key2].masking] = True

        # build self_attention mask masking for spatial query
        # spatial query attend with itself
        if 'queries_spatial' in self_attn_name and 'tokens_spatial' in self_attn_name:
            diag_mask = ~(torch.eye(self.extra['spatial_query_number']).repeat_interleave(self.extra['sample_size'],dim=0).repeat_interleave(self.extra['sample_size'],dim=1)).bool()
            self_attn_mask[:,self.query_index['queries_spatial'][0]:self.query_index['queries_spatial'][1],self.query_index['queries_spatial'][0]:self.query_index['queries_spatial'][1]] = diag_mask[None,]
            # spatial query attend with spatial token
            indices = self.extra['spatial_indices'].permute(0,2,1)
            diag_index = torch.arange(self.extra['spatial_query_number'], device=indices.device).repeat_interleave(self.extra['sample_size'],dim=0)[None,:,None]
            diag_mask = ~(indices == diag_index)
            self_attn_mask[:,self.query_index['queries_spatial'][0]:self.query_index['queries_spatial'][1],self.query_index['tokens_spatial'][0]:self.query_index['tokens_spatial'][1]] = diag_mask
            # spatial token attend with itself
            diag_mask = ~(indices == indices.transpose(1,2))
            self_attn_mask[:,self.query_index['tokens_spatial'][0]:self.query_index['tokens_spatial'][1],self.query_index['tokens_spatial'][0]:self.query_index['tokens_spatial'][1]] = diag_mask

        if 'memory_indices' in self.extra:
            # spatial query attend with memory
            memory_indices = self.extra['memory_indices'][None,None,:]
            diag_index = torch.arange(self.extra['spatial_query_number'], device=memory_indices.device).repeat_interleave(self.extra['sample_size'],dim=0)[None,:,None]
            diag_mask = ~(diag_index == memory_indices)
            self_attn_mask[:,self.query_index['queries_spatial'][0]:self.query_index['queries_spatial'][1],self.query_index['memories_spatial'][0]:self.query_index['memories_spatial'][1]] = diag_mask
            # memory attend with itself
            diag_mask = ~(memory_indices == memory_indices.transpose(1,2))
            self_attn_mask[:,self.query_index['memories_spatial'][0]:self.query_index['memories_spatial'][1],self.query_index['memories_spatial'][0]:self.query_index['memories_spatial'][1]] = diag_mask

        self_attn_mask = self_attn_mask.repeat_interleave(num_heads, dim=0)
        return output, pos_emb, self_attn_mask

    def update_variables(self, output, mode):
        name_set = self.self_attn_name if mode=='self_attn' else self.cross_attn_name
        for key in name_set:
            self.attn_variables[key].output = output[self.query_index[key][0]:self.query_index[key][1]]

    def update_spatial_results(self, results):
        v_emb = results['pred_smaskembs']
        pred_smasks = results['pred_smasks']

        s_emb = results['pred_pspatials']
        diag_mask = ~(torch.eye(self.extra['spatial_query_number'], device=s_emb.device).repeat_interleave(self.extra['sample_size'],dim=0)).bool()
        offset = torch.zeros_like(diag_mask, device=s_emb.device).float()
        offset.masked_fill_(diag_mask, float("-inf"))

        pred_logits = v_emb @ s_emb.transpose(1,2) + offset[None,]
        bs,_,ns=pred_logits.shape
        _,_,h,w=pred_smasks.shape        

        logits_idx_y = pred_logits.max(dim=1)[1]
        logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)[:,None].repeat(1, logits_idx_y.shape[1])
        logits_idx = torch.stack([logits_idx_x, logits_idx_y]).view(2,-1).tolist()
        pred_masks_pos = pred_smasks[logits_idx].reshape(bs,ns,h,w)
        extra = {"prev_mask": pred_masks_pos}
        return extra

    def organize_output(self, ):
        outputs = {}
        outputs['aux_outputs'] = [{} for i in range(self.num_layers)]
        for key, values in self.output.items():
            for _key, idx_name in zip(predict_name_matcher[key], predict_index_matcher[key]):
                if idx_name not in self.query_index:
                    continue
                outputs[_key] = self.output[key][-1][:,self.query_index[idx_name][0]:self.query_index[idx_name][1]]
                for idx, aux_values in enumerate(self.output[key][:-1]):
                    outputs['aux_outputs'][idx][_key] = aux_values[:,self.query_index[idx_name][0]:self.query_index[idx_name][1]]
        if self.task == 'spatial' or self.task == 'refimg':
            outputs = self.update_spatial_results(outputs)
        # outputs = self.update_spatial_results(outputs)
        return outputs