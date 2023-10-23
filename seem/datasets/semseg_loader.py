from PIL import Image
import scipy.io
import numpy as np

def load_semseg(filename, loader_type):
    if loader_type == 'PIL':
        semseg = np.array(Image.open(filename), dtype=np.int)
    elif loader_type == 'MAT':
        semseg = scipy.io.loadmat(filename)['LabelMap']
    return semseg