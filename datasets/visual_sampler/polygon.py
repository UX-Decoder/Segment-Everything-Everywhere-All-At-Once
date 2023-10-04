import random

import numpy as np
import torch
from scipy.special import binom
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)

def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y,a

class Polygon:
    def __init__(self, cfg, is_train):
        self.max_points = cfg['STROKE_SAMPLER']['POLYGON']['MAX_POINTS']
        self.eval_points = cfg['STROKE_SAMPLER']['EVAL']['MAX_ITER']
        self.is_train = is_train

    def get_random_points_from_mask(self, mask, n=3):
        h,w = mask.shape
        view_mask = mask.reshape(h*w)
        non_zero_idx = view_mask.nonzero()[:,0]
        selected_idx = torch.randperm(len(non_zero_idx))[:n]
        non_zero_idx = non_zero_idx[selected_idx]
        y = (non_zero_idx // w)*1.0/(h+1)
        x = (non_zero_idx % w)*1.0/(w+1)
        return torch.cat((x[:,None],y[:,None]), dim=1).numpy()

    def draw(self, mask=None, box=None):
        if mask.sum() < 10:
            return torch.zeros(mask.shape).bool() # if mask is empty
        if not self.is_train:
            return self.draw_eval(mask=mask, box=box)
        # box: x1,y1,x2,y2
        x1,y1,x2,y2 = box.int().unbind()
        rad = 0.2
        edgy = 0.05
        num_points = random.randint(1, min(self.max_points, mask.sum().item()))
        a = self.get_random_points_from_mask(mask[y1:y2,x1:x2], n=num_points)
        x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
        x = x.clip(0.0, 1.0)
        y = y.clip(0.0, 1.0)
        points = torch.from_numpy(np.concatenate((y[None,]*(y2-y1-1).item(),x[None,]*(x2-x1-1).item()))).int()
        canvas = torch.zeros((y2-y1, x2-x1))
        canvas[points.long().tolist()] = 1
        rand_mask = torch.zeros(mask.shape)
        rand_mask[y1:y2,x1:x2] = canvas
        return rand_mask.bool()

    def draw_eval(self, mask=None, box=None):
        # box: x1,y1,x2,y2
        x1,y1,x2,y2 = box.int().unbind()
        rad = 0.2
        edgy = 0.05
        num_points = min(self.eval_points, mask.sum().item())
        a = self.get_random_points_from_mask(mask[y1:y2,x1:x2], n=num_points)
        rand_masks = []
        for i in range(len(a)):
            x,y, _ = get_bezier_curve(a[:i+1],rad=rad, edgy=edgy)
            x = x.clip(0.0, 1.0)
            y = y.clip(0.0, 1.0)
            points = torch.from_numpy(np.concatenate((y[None,]*(y2-y1-1).item(),x[None,]*(x2-x1-1).item()))).int()
            canvas = torch.zeros((y2-y1, x2-x1))
            canvas[points.long().tolist()] = 1
            rand_mask = torch.zeros(mask.shape)
            rand_mask[y1:y2,x1:x2] = canvas

            struct = ndimage.generate_binary_structure(2, 2)
            rand_mask = torch.from_numpy((ndimage.binary_dilation(rand_mask, structure=struct, iterations=5).astype(rand_mask.numpy().dtype)))
            rand_masks += [rand_mask.bool()]
        return torch.stack(rand_masks)

    def __repr__(self,):
        return 'polygon'