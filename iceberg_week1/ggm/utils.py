from functools import reduce, wraps

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, ZeroPadding2D, Dropout

@wraps(Conv2D)
def GnetConv2D(*args, **kwargs):
    gnet_conv_kwargs = {
        'padding' : 'same',
        'data_format' : 'channels_last'
    }
    gnet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **gnet_conv_kwargs)

@wraps(MaxPool2D)
def GnetMaxPool2D(*args, **kwargs):
    gnet_maxp_kwargs = {
        'padding' : 'valid',
        'data_format' : 'channels_last',
        'pool_size' : (2, 2),
        'strides' : 1
    }
    gnet_maxp_kwargs.update(kwargs)
    return MaxPool2D(*args, **gnet_maxp_kwargs)

@wraps(UpSampling2D)
def GnetUpSample2D(*args, **kwargs):
    gnet_up_kwargs = {
        'size' : (1, 1),
        'data_format' : 'channels_last'
    }
    gnet_up_kwargs.update(kwargs)
    return UpSampling2D(*args, **gnet_up_kwargs)

@wraps(ZeroPadding2D)
def GnetZeroPad2D(*args, **kwargs):
    gnet_zerop_kwargs = {
        'data_format' : 'channels_last'
    }
    gnet_zerop_kwargs.update(kwargs)
    return ZeroPadding2D(*args, **gnet_zerop_kwargs)

@wraps(Dropout)
def GnetDropOut(*args, **kwargs):
    gnet_drop_kwargs = {
        'rate' : 0.3
    }
    gnet_drop_kwargs.update(kwargs)
    return Dropout(*args, **gnet_drop_kwargs)

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
