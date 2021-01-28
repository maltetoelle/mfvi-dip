from typing import List, Tuple, Union

import torch
from torch import Tensor

def calc_uncert_sgld(img_list: List[Tensor], reduction: str = 'mean', ale: bool = False) -> Tuple(Union[float, Tensor]):
    img_list = torch.cat(img_list, dim=0)
    epi = torch.var(img_list[:,:-1], dim=0)
    ale = torch.mean(img_list[:,-1:].exp(), dim=0) if ale else 0
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean(), epi.mean(), uncert.mean()
    else:
        return ale, epi, uncert
