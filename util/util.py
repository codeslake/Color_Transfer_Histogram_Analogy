from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect
import re
import numpy as np
import os
import collections
from skimage import color
import fnmatch

def LAB2RGB(I):
    l = I[:, :, 0] / 255.0 * 100.0
    a = I[:, :, 1] / 255.0 * (98.2330538631 + 86.1830297444) - 86.1830297444
    b = I[:, :, 2] / 255.0 * (94.4781222765 + 107.857300207) - 107.857300207

    rgb = color.lab2rgb(np.dstack([l, a, b]).astype(np.float64))
    return rgb

def HSV2RGB(I):

    h = I[:, :, 0] / 255.0 * 360.0
    s = I[:, :, 1] / 255.0 * 100.0
    v = I[:, :, 2] / 255.0 * 100.0

    hsv = color.hsv2rgb(np.dstack([h, s, v]).astype(np.float64))*255
    return hsv

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array

def tensor2im(image_tensor,img_type,imtype=np.uint8):
    image_numpy = image_tensor.detach().cpu().numpy()
    if image_numpy.shape[1] == 1:
        image_numpy = np.tile(image_numpy, (1, 3, 1, 1))
    image_numpy = ((np.transpose(image_numpy, (0, 2, 3, 1)) * 0.5) + 0.5) * 255.0 #-1~1  => 

    # Additional
    image_numpy = image_numpy[0]
    if img_type == 'rgb':
        # image_numpy = np.clip(image_numpy,0,255).astype(np.uint8)
        image_numpy = image_numpy / 255.
        image_numpy = image_numpy * 10.
    elif img_type == 'hsv':
        image_numpy = HSV2RGB(image_numpy) # for Lab to RGB image
        image_numpy = np.clip(image_numpy,0,255)        
    elif img_type == 'lab':
        image_numpy = LAB2RGB(image_numpy) # for Lab to RGB image
        # image_numpy = np.clip(image_numpy,0,255)        
    else:
        print(ERROR)

    # image = torch.from_numpy(image_numpy.transpose(2,0,1)).byte().cuda()
    image = torch.from_numpy(image_numpy.transpose(2,0,1)).cuda()

    return image

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_file_end_with(path, regex):
    file_paths = get_file_path(path, [regex])

    for i in np.arange(len(file_paths)):
        os.remove(file_paths[i])

def get_file_path(path, regex):
    file_path = []
    for root, dirnames, filenames in os.walk(path):
        for i in np.arange(len(regex)):
            for filename in fnmatch.filter(filenames, regex[i]):
                file_path.append(os.path.join(root, filename))

    return file_path
