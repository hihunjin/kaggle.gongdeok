import os
from utils import *
from pprint import pprint

from configparser import ConfigParser

import tensorflow as tf
from tensorflow.keras import Model

def parse_config(config, presets):
    presets['dataset_path'] = config.get('basic', 'dataset_path')
    presets['save_path'] = config.get('basic', 'save_path')
    presets['prefix_checkpoint'] = config.get('basic', 'prefix_checkpoint')
    
    presets['max_epoch'] = config.getint('training', 'max_epoch')
    presets['batch_size'] = config.getint('training', 'batch_size')
    presets['filters'] = list(map(int, config.get('training', 'filters').split(',')))
    presets['kernel_size'] = config.getint('training', 'kernel_size')
    
    presets['flip'] = config.getint('data','flip')
    presets['brightness'] = config.getint('data','brightness')
    presets['saturation'] = config.getint('data','saturation')
    presets['hue'] = config.getint('data','hue')
    presets['contrast'] = config.getint('data','contrast')
    return presets

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
            print('-A config_ae.ini file loacted.')

        config = ConfigParser()
        config.read(self.config_file)
        self.presets = parse_config(config, self.defaults)
        pprint(self.presets)
        self.build_base()

    def build_base(self):
        self.encod = self.encoder()
        self.decod = self.decoder()
        self.models = compose(
            self.encod,
            self.decod
        )

    def saved_model(self):
        self.encod.save(
            self.presets['save_path']
        )

    def load_model(self):
        self.encod = tf.keras.models.load_model(
            self.presets['save_path']
        )
        return self.encod

    @tf.function
    def call(self, image):
        return self.models(image)

    def encoder(self):
        filters, kernel_size = self.presets['filters'], self.presets['kernel_size']
        encoder = tf.keras.Sequential([
            GnetConv2D(filters[0], kernel_size, input_shape = (75,75,3)),
            GnetMaxPool2D(),
            GnetDropOut(rate = 0.2),
            GnetConv2D(filters[1], kernel_size),
            GnetMaxPool2D(),
            GnetDropOut(rate = 0.2),
            GnetConv2D(filters[2], kernel_size),
            GnetMaxPool2D()
        ])
        return encoder

    def decoder(self):
        filters, kernel_size = self.presets['filters'], self.presets['kernel_size']
        decoder = tf.keras.Sequential([
            GnetUpSample2D(),
            GnetConv2D(filters[2], kernel_size),
            GnetUpSample2D(),
            GnetZeroPad2D(padding = (1,1)),
            GnetConv2D(filters[1], kernel_size),
            GnetUpSample2D(),
            GnetZeroPad2D(padding = ((1,0),(1,0))),
            GnetConv2D(filters[0], kernel_size),
            GnetConv2D(3, kernel_size, activation=None)
        ])
        return decoder

if __name__ == "__main__":
    model = AutoEncoder()
    
