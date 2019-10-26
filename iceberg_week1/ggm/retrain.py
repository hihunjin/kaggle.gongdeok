import os
import numpy as np
import matplotlib.pyplot as plt

from model import AutoEncoder
from dataloader import Dataloader

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

if __name__ == "__main__":
    show = False

    AE = AutoEncoder()
    print('loading....', end ='\r')
    encoder = AE.load_model()
    print('loaded.')

    gmodel = tf.keras.Sequential([
      encoder,
      Flatten(),
      Dense(256),
      Activation('selu'),
      Dropout(0.2),
      Dense(128),
      Activation('selu'),
      Dropout(0.2),
      Dense(1),
      Activation('sigmoid')
    ])

    mypotim=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()


    data_helper = Dataloader(
        datapath = AE.presets['dataset_path'],
        normalize= True,
        gaussian = True # if False bionomial
    )

    augment_kwargs = {
        'flip' : AE.presets['flip'],
        'brightness' : AE.presets['brightness'],
        'saturation' : AE.presets['saturation'],
        'hue' : AE.presets['hue'],
        'contrast' : AE.presets['contrast']
    }

    train_ds, valid_ds = data_helper.dataload(
        train= True,
        at_encoder= False,
        train_size= 0.8,
        batch_size= AE.presets['batch_size'],
        **augment_kwargs
    )

    train_ge = tf.compat.v1.data.make_one_shot_iterator(train_ds)
    valid_ge = tf.compat.v1.data.make_one_shot_iterator(valid_ds)

    gmodel.fit_generator(
      train_ge,
      steps_per_epoch=400,
      epochs = 100,
      verbose = 1,
      validation_data=valid_ge,
      validation_steps=20
    )


    gmodel.save('./result2')