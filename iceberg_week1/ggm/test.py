import os
import numpy as np
import matplotlib.pyplot as plt

from model import AutoEncoder
from dataloader import Dataloader

import tensorflow as tf
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

plt.style.use('dark_background')

if __name__ == "__main__":
  show = True

  AE = AutoEncoder()
  encoder = AE.load_model()

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
    at_encoder= True,
    train_size= 0.8,
    batch_size= AE.presets['batch_size'],
    **augment_kwargs
  )

  for v_inputs, v_outputs in valid_ds:
    v_predictions = AE(v_inputs)
 
    if show:
      plt.ioff()
      fig = plt.figure(figsize = (18, 9))

      plt.subplot(1, 3, 1)
      plt.xlabel('noise input')
      plt.imshow(v_inputs[0][:,:,0].numpy())
      plt.subplot(1, 3, 2)
      plt.xlabel('prediction')
      plt.imshow(v_predictions[0][:,:,0].numpy())
      plt.subplot(1, 3, 3)
      plt.xlabel('real input')
      plt.imshow(v_outputs[0][:,:,0].numpy())
  
      plt.draw()
      plt.pause(1) # Linux Cent OS 에서는 그림 갱신을 위해 필수 입력.
      plt.ion()
      plt.show()
      plt.pause(1)
