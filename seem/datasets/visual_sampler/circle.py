import random
import torch

from .mask_generators import get_mask_by_input_strokes

class Circle:
    def __init__(self, cfg, is_train=True):
        self.num_stroke = cfg['STROKE_SAMPLER']['CIRCLE']['NUM_STROKES']
        self.stroke_preset = cfg['STROKE_SAMPLER']['CIRCLE']['STROKE_PRESET']
        self.stroke_prob = cfg['STROKE_SAMPLER']['CIRCLE']['STROKE_PROB']
        self.max_eval = cfg['STROKE_SAMPLER']['EVAL']['MAX_ITER']
        self.is_train = is_train

    @staticmethod
    def get_stroke_preset(stroke_preset):
        if stroke_preset == 'object_like':
            return {
                "nVertexBound": [5, 30],
                "maxHeadSpeed": 15,
                "maxHeadAcceleration": (10, 1.5),
                "brushWidthBound": (20, 50),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 10,
                "maxLineAcceleration": (5, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 10,
            }
        elif stroke_preset == 'object_like_middle':
            return {
                "nVertexBound": [5, 15],
                "maxHeadSpeed": 8,
                "maxHeadAcceleration": (4, 1.5),
                "brushWidthBound": (20, 50),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 5,
                "maxLineAcceleration": (5, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 10,
            }
        elif stroke_preset == 'object_like_small':
            return {
                "nVertexBound": [5, 20],
                "maxHeadSpeed": 7,
                "maxHeadAcceleration": (3.5, 1.5),
                "brushWidthBound": (10, 30),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 5,
                "maxLineAcceleration": (3, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 4,
            }
        else:
            raise NotImplementedError(f'The stroke presetting "{stroke_preset}" does not exist.')

    def get_random_points_from_mask(self, mask, n=5):
        h,w = mask.shape
        view_mask = mask.reshape(h*w)
        non_zero_idx = view_mask.nonzero()[:,0]
        selected_idx = torch.randperm(len(non_zero_idx))[:n]
        non_zero_idx = non_zero_idx[selected_idx]
        y = (non_zero_idx // w)*1.0
        x = (non_zero_idx % w)*1.0
        return torch.cat((x[:,None], y[:,None]), dim=1).numpy()

    def draw(self, mask=None, box=None):
        if mask.sum() < 10: # if mask is nearly empty
            return torch.zeros(mask.shape).bool()
        if not self.is_train:
            return self.draw_eval(mask=mask, box=box)
        stroke_preset_name = random.choices(self.stroke_preset, weights=self.stroke_prob, k=1)[0] # select which kind of object to use
        preset = Circle.get_stroke_preset(stroke_preset_name)
        nStroke = min(random.randint(1, self.num_stroke), mask.sum().item())
        h,w = mask.shape
        points = self.get_random_points_from_mask(mask, n=nStroke)
        rand_mask = get_mask_by_input_strokes(
            init_points=points,
            imageWidth=w, imageHeight=h, nStroke=min(nStroke, len(points)), **preset)
        rand_mask = (~torch.from_numpy(rand_mask)) * mask
        return rand_mask

    def draw_eval(self, mask=None, box=None):
        stroke_preset_name = random.choices(self.stroke_preset, weights=self.stroke_prob, k=1)[0] # select which kind of object to use
        preset = Circle.get_stroke_preset(stroke_preset_name)
        nStroke = min(self.max_eval, mask.sum().item())
        h,w = mask.shape
        points = self.get_random_points_from_mask(mask, n=nStroke)
        rand_masks = []
        for i in range(len(points)):
            rand_mask = get_mask_by_input_strokes(
                init_points=points[:i+1],
                imageWidth=w, imageHeight=h, nStroke=min(nStroke, len(points[:i+1])), **preset)
            rand_masks += [(~torch.from_numpy(rand_mask)) * mask]
        return torch.stack(rand_masks)

    @staticmethod
    def draw_by_points(points, mask, h, w):
        stroke_preset_name = random.choices(['object_like', 'object_like_middle', 'object_like_small'], weights=[0.33,0.33,0.33], k=1)[0] # select which kind of object to use
        preset = Circle.get_stroke_preset(stroke_preset_name)
        rand_mask = get_mask_by_input_strokes(
            init_points=points,
            imageWidth=w, imageHeight=h, nStroke=len(points), **preset)[None,]
        rand_masks = (~torch.from_numpy(rand_mask)) * mask
        return rand_masks

    def __repr__(self,):
        return 'circle'