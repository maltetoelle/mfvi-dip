import torch
import torch.nn as nn
from .common import *

def skip(
        num_input_channels=2,
        num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3,
        filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad='zero',
        upsample_mode='nearest',
        downsample_mode='stride',
        act_fun='LeakyReLU',
        need1x1_up=True,
        ffg=False, implicit=False,
        dropout_mode_down='None', dropout_p_down=0.5,
        dropout_mode_up='None', dropout_p_up=0.5,
        dropout_mode_skip='None', dropout_p_skip=0.5,
        dropout_mode_output='None', dropout_p_output=0.5):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """

    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels

    if isinstance(act_fun, nn.Module):
        name_act_fun = act_fun._get_name()
    else:
        name_act_fun = act_fun

    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size,
                          bias=need_bias, pad=pad,
                          dropout_mode=dropout_mode_skip, dropout_p=dropout_p_skip, ffg=ffg, implicit=implicit, name="skip_%d" % i))
            skip.add(bn(num_channels_skip[i]))

            skip.add(act(act_fun))
            if dropout_mode_skip == 'afterrelu':
                skip.add(nn.Dropout2d(p=dropout_p_skip))

        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i], dropout_mode=dropout_mode_down, dropout_p=dropout_p_down, ffg=ffg, implicit=implicit, name="down_%d_1" % i))

        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if dropout_mode_down == 'afterrelu':
            deeper.add(nn.Dropout2d(p=dropout_p_down))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad,
                        dropout_mode=dropout_mode_down, dropout_p=dropout_p_down, ffg=ffg, implicit=implicit, name="down_%d_2" % i))

        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        if dropout_mode_down == 'afterrelu':
            deeper.add(nn.Dropout2d(p=dropout_p_down))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad,
                           dropout_mode=dropout_mode_up, dropout_p=dropout_p_up, ffg=ffg, implicit=implicit, name="up_%d_1" % i))

        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
        if dropout_mode_up == 'afterrelu':
            model_tmp.add(nn.Dropout2d(p=dropout_p_up))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad, dropout_mode=dropout_mode_up, dropout_p=dropout_p_up, ffg=ffg, implicit=implicit, name="down_%d_2" % i))

            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))
            if dropout_mode_up == 'afterrelu':
                model_tmp.add(nn.Dropout2d(p=dropout_p_up))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad, dropout_mode=dropout_mode_output, dropout_p=dropout_p_output, ffg=ffg, implicit=implicit, name="out_%d" % i))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
