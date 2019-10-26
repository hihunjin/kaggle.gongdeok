import numpy as np
import matplotlib.pyplot as plt

from model import AutoEncoder
from dataloader import Dataloader

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

plt.style.use('dark_background')

@tf.function
def train_step(inputs, outputs): # shape = (16, 75, 75, 2)
    with tf.GradientTape() as tape:
        predictions = AE(inputs)
        loss = loss_object(predictions, outputs)

    gradients = tape.gradient(loss, AE.trainable_variables)
    optimizer.apply_gradients(zip(gradients, AE.trainable_variables))

    return loss, predictions

@tf.function
def valid_step(inputs, outputs):
    predictions = AE(inputs)
    loss = loss_object(outputs, predictions)

    return loss, predictions

def train_init(lr_rate):
    return tf.keras.losses.mean_squared_error, tf.keras.optimizers.Adam(learning_rate=lr_rate)


if __name__ == "__main__":
    show = False

    AE = AutoEncoder()

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

    loss_object, optimizer = train_init(lr_rate = 0.0001)

    train_ds, valid_ds = data_helper.dataload(
        train= True,
        at_encoder= True,
        train_size= 0.8,
        batch_size= AE.presets['batch_size'],
        **augment_kwargs
    )

    for epoch in range(AE.presets['max_epoch']):
        for inputs, outputs in train_ds:
            loss, predictions = train_step(inputs, outputs)
        train_loss = tf.reduce_mean(tf.reduce_mean(loss, axis=0))

        for v_inputs, v_outputs in valid_ds:
            v_loss, v_predictions = valid_step(v_inputs, v_outputs)
        valid_loss = tf.reduce_mean(tf.reduce_mean(v_loss, axis=0))

        template = 'epoch: {:<10} train: {:<30} valid: {:<30}'
        print (template.format(epoch+1, train_loss, valid_loss))
        
        if epoch and show:
            plt.close()

        if show:
            plt.ioff()
            fig = plt.figure(epoch, figsize = (18, 9))
        
            plt.subplot(2, 3, 1)
            plt.xlabel('noise input')
            plt.imshow(inputs[0][:,:,0].numpy())
            plt.subplot(2, 3, 2)
            plt.xlabel('real input')
            plt.imshow(outputs[0][:,:,0].numpy())
            plt.subplot(2, 3, 3)
            plt.xlabel('predictions')
            plt.imshow(predictions[0][:,:,0].numpy())
            plt.subplot(2, 3, 4)
            plt.xlabel('valid input')
            plt.imshow(v_inputs[0][:,:,0].numpy())
            plt.subplot(2, 3, 5)
            plt.xlabel('valid input')
            plt.imshow(v_outputs[0][:,:,0].numpy())
            plt.subplot(2, 3, 6)
            plt.xlabel('predictions')
            plt.imshow(v_predictions[0][:,:,0].numpy())
            plt.draw()
            plt.pause(1) # Linux Cent OS 에서는 그림 갱신을 위해 필수 입력.
            plt.ion()
            plt.show()
            plt.pause(1)

    AE.saved_model()