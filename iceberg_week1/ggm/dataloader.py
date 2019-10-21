import os
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from sklearn.model_selection import train_test_split

class Dataloader(object):
    def __init__(self, datapath, normalize = True, gaussian = True):
        self.train = pd.read_json(datapath + '/train.json')
        self.test = pd.read_json(datapath + '/test.json')
        self.vector2image()
        self.normalize() if normalize else None
        self.noise = True

    def getvalue(self, key):
        return self.train[key] if key == 'is_iceberg' else self.train[key], self.test[key]

    def vector2image(self): # 3 channels.. make
        X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in self.train["band_1"]])
        X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in self.train["band_2"]])

        self.x_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)

        X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in self.test["band_1"]])
        X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in self.test["band_2"]])

        self.x_test = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
        
        X_band_1, X_band_2 = [], [] # flush
        
    def normalize(self):
        self.x_train = (self.x_train - np.mean(self.x_train)) / np.std(self.x_train)
        self.x_test = (self.x_test - np.mean(self.x_test)) / np.std(self.x_test)

    # def noise(self, gaussian = True):
    #     if gaussian:
    #         self.x_train = self.x_train + np.random.normal(0, 1, size = self.x_train.shape)
    #     else: # binomial
    #         self.x_train = self.x_train * np.random.binomial(1, 0.8, size = self.x_train.shape)

    def augment_image(self, augmented, kwargs):
        """ crop은.. 나중에 """
        # print(augmented)
        
        if kwargs['flip']:
            augmented = tf.image.random_flip_left_right(image = augmented)
            augmented = tf.image.random_flip_up_down(image = augmented)
        if kwargs['brightness']:
            augmented = tf.image.random_brightness(image = augmented, max_delta=32./255.)
        if kwargs['saturation']: # 채도
            augmented = tf.image.random_saturation(image = augmented, lower=0.5, upper=1.5)
        if kwargs['hue']: # 색조
            augmented = tf.image.random_hue(image = augmented, max_delta=0.2)
        if kwargs['contrast']: # 대조
            augmented = tf.image.random_contrast(image = augmented, lower=0.5, upper=1.5)

        origin = augmented

        if self.noise:
            augmented = augmented + tf.random.normal(mean = 0, stddev = .2, shape = augmented.shape, dtype = tf.float32)

        print(id(augmented))
        print(id(origin))
        return augmented, origin
    
    def dataload(self, train = True, at_encoder = True, train_size = 0.8, batch_size = 16, *args, **kwargs):
        if train:
            if at_encoder:
                x_train, x_valid, y_train, y_valid = train_test_split(self.x_train, self.x_train, random_state = 1, train_size = 0.8)
                
                train_ds = tf.data.Dataset.from_tensor_slices(x_train).map(lambda elm: self.augment_image(augmented = elm, kwargs = kwargs)).shuffle(10000).batch(batch_size)
                # train_ds = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(10000).batch(batch_size)
                valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).shuffle(10000).batch(batch_size)
                # augment_helper = partial(self.augment_image, **kwargs)
                return train_ds, valid_ds
            else:
                x_train, x_valid, y_train, y_valid = train_test_split(self.x_train, self.train['is_iceberg'], random_state = 1, train_size = 0.8)
                train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
                valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).shuffle(10000).batch(batch_size)
                return train_ds.map(lambda elm : self.augment_image(elm, **kwargs)), valid_ds
        return self.x_test