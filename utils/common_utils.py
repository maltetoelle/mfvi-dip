import torch
import torch.nn as nn
import torchvision
import sys
# import tensorflow as tf

import numpy as np
from PIL import Image
import PIL
import numpy as np
import math
# import cv2
import io
from collections import OrderedDict
import subprocess
import re
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

def get_fname(img_name):
    if img_name == 'xray':
        fname = 'data/bayesian/BACTERIA-1351146-0006.jpg'
    if img_name == 'xray2':
        fname = 'data/bayesian/VIRUS-9815549-0001.png'
    elif img_name == 'oct':
        fname = 'data/bayesian/CNV-9997680-30.png'
    elif img_name == 'oct2':
        fname = 'data/bayesian/CNV-7902439-111.png'
    elif img_name == 'us':
        fname = 'data/bayesian/081_HC.jpg'
    elif img_name == 'us2':
        fname = 'data/bayesian/196_HC.png'

    elif img_name == 'mri0':
        # fname = 'data/sr/MRI/img_203.png'
        fname = 'data/sr/MRI/img_203_res.png'
    elif img_name == 'mri1':
        # fname = 'data/sr/MRI/img_139.png'
        fname = 'data/sr/MRI/img_139_res.png'
    elif img_name == 'mri2':
        fname = 'data/sr/MRI/img_147.png'
    elif img_name == 'mri3':
        fname = 'data/sr/MRI/img_153.png'
    elif img_name == 'mri4':
        fname = 'data/sr/MRI/img_229.png'
    elif img_name == 'mri5':
        fname = 'data/sr/MRI/img_255.png'

    elif img_name == 'ct0':
        fname = 'data/sr/CT/ct0_res.png'
    elif img_name == 'ct1':
        fname = 'data/sr/CT/ct1_res.png'
    elif img_name == 'ct2':
        fname = 'data/sr/CT/ct2.png'
    elif img_name == 'ct3':
        fname = 'data/sr/CT/ct3.png'
    elif img_name == 'ct4':
        fname = 'data/sr/CT/ct4.png'
    elif img_name == 'ct5':
        fname = 'data/sr/CT/ct5.png'

    elif img_name == 'peppers':
        fname = 'GP_DIP/data/denoising/Dataset/image_Peppers512rgb.png'
    elif img_name == 'zebra':
        fname = 'data/sr/zebra_GT.png'
    elif img_name == 'library':
        fname = 'data/inpainting/library.png'

    elif img_name == "skin_lesion0":
        fname = "data/inpainting/skin_lesions/hair_0_res.png"
    elif img_name == "skin_lesion1":
        fname = "data/inpainting/skin_lesions/hair_1_res.png"
    elif img_name == "skin_lesion2":
        fname = "data/inpainting/skin_lesions/hair_2_res.png"
    elif img_name == "skin_lesion3":
        fname = "data/inpainting/skin_lesions/hair_3_res.png"
    elif img_name == "skin_lesion4":
        fname = "data/inpainting/skin_lesions/hair_4_res.png"
    elif img_name == "skin_lesion5":
        fname = "data/inpainting/skin_lesions/hair_5_res.png"

    return fname

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2),
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos', opt='RGB'):
    """Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)

    plt.figure(figsize=(len(images_np) + factor, 12 + factor))

    if images_np[0].shape[0] == 1:
        cmap = 'gray' if opt != 'map' else cm.RdYlGn
        plt.imshow(grid[0], cmap=cmap, interpolation=interpolation)

    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.show()

    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10, library='torch', data_format='channels_first'):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]

        net_input = torch.zeros(shape)

        if noise_type == 'u':
            net_input.uniform_()
        elif noise_type == 'n':
            net_input.normal_()
        else:
            assert False

        net_input *= var

    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])

        net_input =  np_to_torch(meshgrid)

    else:
        assert False

    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def check_gpu(device_no: int = 1):
    _output = subprocess.check_output(['nvidia-smi', '-q', '-d', 'temperature,power', '-i', str(device_no)])
    _output = str(_output).split('\\n')

    output = {}
    suffix = ''
    for o in _output:
        _o = o.split(':')

        if len(_o) != 2:
            if _o[0].strip() == 'Power Samples':
                suffix = ' Power'
            continue
        try:
            output[_o[0].strip() + suffix] = float(re.compile(r'[^\d.]+').sub("", _o[1]))
        except:
            continue

    return output

def fig_to_np(fig, dpi=180):
    buf = io.BytesIO()
    # import pdb;pdb.set_trace()
    if isinstance(fig, matplotlib.image.AxesImage):
        fig = fig.get_figure()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_pil = Image.open(buf)
    # img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    # buf.close()
    # img = pil_to_np(img_pil)
    # img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # remove 4th channel (transparency)
    img = np.array(img_pil.getdata()).astype(np.float32)[...,:3].reshape((img_pil.size[0],img_pil.size[1],3))

    return img

def plot_feature_maps(hook):
    ds = []
    for i in range(1, hook.shape[0]):
        if hook.shape[0] % i == 0:
            ds.append([i, hook.shape[0] / i])

    ds = ds[np.argmin(np.abs(np.diff(np.array(ds), axis=1)))]
    bd, od = int(max(ds)), int(min(ds))

    fig, axs = plt.subplots(od, bd, constrained_layout=True)
    plt.axis('off')

    for i in range(hook.shape[0]):
        ax = axs[int(i / bd), i % bd]
        ax.imshow(hook[i])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

    fig_np = fig_to_np(fig)
    return fig_np

def init_normal(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.normal_(m.weight, mean=0, std=0.1)

def init_uniform(m):
    if not isinstance(m, nn.BatchNorm2d) and hasattr(m, 'weight'):
        m.weight.data.uniform_(-0.1, 0.1)
        if m.bias is not None:
            m.bias.data.uniform_(-0.1, 0.1)

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
