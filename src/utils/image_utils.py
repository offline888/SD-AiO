import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange

def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
def rgb2bgr(im): return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def imread(path, chn='rgb', dtype='float32'):
    """
    Read image.
    chn: 'rgb', 'bgr' or 'gray'
    Returns: h x w x c, numpy array
    """
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    try:
        if chn.lower() == 'rgb':
            if im.ndim == 3:
                im = bgr2rgb(im)
            else:
                im = np.stack((im, im, im), axis=2)
        elif chn.lower() == 'gray':
            assert im.ndim == 2
    except Exception:
        print(str(path))

    if dtype == 'float32':
        return im.astype(np.float32) / 255.
    elif dtype == 'float64':
        return im.astype(np.float64) / 255.
    elif dtype == 'uint8':
        return im
    else:
        sys.exit('dtype must be float32, float64 or uint8')


def img2tensor(imgs, out_type=torch.float32):
    """
    Convert image numpy arrays into torch tensors.
    Accepts:
        - 3D numpy array (H x W x C)
        - 2D numpy array (H x W)
        - list of numpy arrays
    Returns:
        - 4D tensor (1 x C x H x W) for single image
        - list of 4D tensors for list input
    """
    def _img2tensor(img):
        if img.ndim == 2:
            return torch.from_numpy(img[None, None]).type(out_type)
        elif img.ndim == 3:
            return torch.from_numpy(rearrange(img, 'h w c -> c h w')).type(out_type).unsqueeze(0)
        raise TypeError(f'2D or 3D numpy array expected, got {img.ndim}D')

    if isinstance(imgs, np.ndarray):
        return _img2tensor(imgs)
    elif isinstance(imgs, list):
        return [_img2tensor(x) for x in imgs]
    raise TypeError(f'Numpy array or list expected, got {type(imgs).__name__}')
