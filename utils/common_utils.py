from typing import List, Union, Tuple

import torch
import torch.nn as nn
import torchvision

import numpy as np
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_fname(img_name: str) -> str:
    """Convenience fn. for getting image by its abbreviation.

    Args:
        img_name: abbreviation of image
    """

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

    elif img_name == 'mri':
        fname = 'data/sr/MRI/img_203_res.png'
    elif img_name == 'mri2':
        fname = 'data/sr/MRI/img_139_res384.png'

    elif img_name == 'ct':
        fname = 'data/sr/CT/ct0_res.png'
    elif img_name == 'ct2':
        fname = 'data/sr/CT/ct1_res.png'

    elif img_name == 'peppers':
        fname = 'GP_DIP/data/denoising/Dataset/image_Peppers512rgb.png'
    elif img_name == 'zebra':
        fname = 'data/sr/zebra_GT.png'
    elif img_name == 'library':
        fname = 'data/inpainting/library.png'

    elif img_name == "skin_lesion":
        fname = "data/inpainting/skin_lesions/hair_0_res.png"
    elif img_name == "skin_lesion2":
        fname = "data/inpainting/skin_lesions/hair_1_res.png"

    return fname

def crop_image(img: np.ndarray, d: int = 32) -> np.ndarray:
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

def get_params(opt_over: List[str],
               net: nn.Module,
               net_input: torch.Tensor,
               downsampler: nn.Module = None) -> List[torch.Tensor]:
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

def get_image_grid(images_np: List[np.ndarray], nrow: int = 8) -> np.ndarray:
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()

def plot_image_grid(images_np: List[np.ndarray],
                    nrow: int = 8,
                    factor: int = 1,
                    interpolation: str = 'lanczos',
                    opt: str = 'RGB') -> np.ndarray:
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

def load(path: str) -> Image:
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path: str, imsize: Union[Tuple[int], int] = -1) -> Tuple[np.ndarray]:
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

def fill_noise(x: torch.Tensor, noise_type: str):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

def get_noise(input_depth: int,
              method: str,
              spatial_size: int,
              noise_type: str = 'u',
              var: float = 1./10) -> torch.Tensor:
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

def pil_to_np(img_PIL: Image) -> np.ndarray:
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np: np.ndarray) -> Image:
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np: np.ndarray) -> torch.Tensor:
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var: torch.Tensor) -> np.ndarray:
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def init_normal(m: nn.Module):
    """Init all weight layers with normal distribution"""
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.1)

def weight_reset(m: nn.Module):
    """"Reset weights to initial values"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
