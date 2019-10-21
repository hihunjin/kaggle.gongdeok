import os
from pprint import pprint
from functools import reduce
from configparser import ConfigParser

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, ZeroPadding2D

def parse_config(config, presets):
    # -------- basic ---------
    presets['dataset_path'] = config.get('basic', 'dataset_path')
    presets['save_path'] = config.get('basic', 'save_path')
    presets['prefix_checkpoint'] = config.get('basic', 'prefix_checkpoint')
    # -------- training ------
    presets['max_epoch'] = config.getint('training', 'max_epoch')
    presets['batch_size'] = config.getint('training', 'batch_size')
    presets['filters'] = list(map(int, config.get('training', 'filters').split(',')))
    presets['kernel_size'] = config.getint('training', 'kernel_size')
    # -------- data ----------
    presets['flip'] = config.getint('data','flip')
    presets['brightness'] = config.getint('data','brightness')
    presets['saturation'] = config.getint('data','saturation')
    presets['hue'] = config.getint('data','hue')
    presets['contrast'] = config.getint('data','contrast')
    return presets


def conv_2d(filter_num, kernel_size):
    return Conv2D(
        filters = filter_num,
        kernel_size = kernel_size,
        padding = 'same',
        activation = 'selu',
        data_format = 'channels_last'
    )


def maxpool_2d():
    return MaxPool2D(
        pool_size = (2, 2),
        strides = 1,
        padding = 'valid',
        data_format = 'channels_last'
    )


def upsample_2d():
    return UpSampling2D(
        size= (1, 1),
        data_format= 'channels_last'
    )

def zeropad_2d(size):
    return ZeroPadding2D(
        padding = size,
        data_format = 'channels_last'
    )


def compose(*funcs): # compose: 구성, 혼합 # network 모델들을 합친다...
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


class AutoEncoder(Model):
    defaults = {
        'dataset_path': './data/iceberg',
        'save_path': './result/iceberg',
        'prefix_checkpoint': 'this-result',
        'max_epoch': 1000,
        'batch_size': 1,
        'filters': [64, 128, 256],
        'kernel_size': 5,
        'flip' : True,
        'brightness' : True,
        'saturation' : True,
        'hue' : True,
        'contrast' : True
    }

    config_file = 'config_ae.ini'

    def __init__(self):
        super(AutoEncoder, self).__init__()

        if os.path.exists(self.config_file):
            print('-A config.ini file loacted.')

        config = ConfigParser()
        config.read(self.config_file)
        self.presets = parse_config(config, self.defaults)
        pprint(self.presets)
        self.encoder()
        self.decoder()

    def call(self, image):
        return compose(
            self.conv_1,
            self.maxpool_1,
            self.conv_2,
            self.maxpool_2,
            self.conv_3,
            self.maxpool_3,
            self.upsample_4,
            self.conv_4,
            self.upsample_5,
            self.padding_5,
            self.conv_5,
            self.upsample_6,
            self.padding_6,
            self.conv_6,
            self.decode
        )(image)

    def encoder(self):
        filters, kernel_size = self.presets['filters'], self.presets['kernel_size']

        # 4D tensor with shape: (samples, rows, cols, channels)
        self.conv_1 = conv_2d(filters[0], kernel_size)
        self.maxpool_1 = maxpool_2d()

        self.conv_2 = conv_2d(filters[1], kernel_size)
        self.maxpool_2 = maxpool_2d()

        self.conv_3 = conv_2d(filters[2], kernel_size)
        self.maxpool_3 = maxpool_2d()

    def decoder(self):
        filters, kernel_size = self.presets['filters'], self.presets['kernel_size']

        self.upsample_4 = upsample_2d()
        self.conv_4 = conv_2d(filters[2], kernel_size)

        self.upsample_5 = upsample_2d()
        self.padding_5 = zeropad_2d(size = (1,1))
        self.conv_5 = conv_2d(filters[1], kernel_size)

        self.upsample_6 = upsample_2d()
        self.padding_6 = zeropad_2d(size = ((1,0),(1,0)))
        self.conv_6 = conv_2d(filters[0], kernel_size)

        self.decode = conv_2d(3, kernel_size)


if __name__ == "__main__":
    model = AutoEncoder()
    
