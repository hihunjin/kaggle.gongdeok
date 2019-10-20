import tensorflow as tf 
import tensorflow.keras.layers as L


class SimpleConvNet(tf.keras.Model):
    
    def __init__(self, size=4, use_inc_angle=False,**kwargs):
        super(SimpleConvNet, self).__init__(**kwargs)

        self.use_inc_angle = use_inc_angle

        self.conv1 = tf.keras.Sequential([
            L.Conv2D(size*16, kernel_size=(3, 3),activation='relu'),
            L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            L.Dropout(0.2)
        ])
        
        self.conv2 = tf.keras.Sequential([
            L.Conv2D(size*32, kernel_size=(3, 3), activation='relu'),
            L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            L.Dropout(0.2)
        ])
        
        self.conv3 = tf.keras.Sequential([
            L.Conv2D(size*32, kernel_size=(3, 3), activation='relu'),
            L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            L.Dropout(0.3)  
        ])
        
        self.conv4 = tf.keras.Sequential([
            L.Conv2D(size*16, kernel_size=(3, 3), activation='relu'),
            L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            L.Dropout(0.3)
        ])
   
        # You must flatten the data for the dense layers
        self.flatten = L.Flatten()

        self.dense1 = tf.keras.Sequential([
            L.Dense(size*128, activation='relu'),
            L.Dropout(0.2)
        ])

        self.dense2 = tf.keras.Sequential([
            L.Dense(size*64, activation='relu'),
            L.Dropout(0.2)
        ])

        self.model_output = L.Dense(1, activation="sigmoid")
        pass 
        
    @tf.function
    def call(self, x, inc_angle=False, training=False):


        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)

        x = self.flatten(x)        

        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)

        return self.model_output(x)


class MultiTaperConvNet(tf.keras.Model):
    
    def __init__(self, size=4,**kwargs):
        super(MultiTaperConvNet, self).__init__(**kwargs)

        self.conv1 = tf.keras.Sequential([
            L.Conv2D(size*16, kernel_size=(5, 5),activation='relu'),
            L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            L.Dropout(0.2),
            L.Conv2D(size*16, kernel_size=(3, 3),activation='relu'),
            L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            L.Dropout(0.2),
            L.Flatten()
        ])
        
        self.conv2 = tf.keras.Sequential([
            L.Conv2D(size*16, kernel_size=(10, 10),activation='relu'),
            L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            L.Dropout(0.2),
            L.Conv2D(size*16, kernel_size=(10, 10),activation='relu'),
            L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            L.Dropout(0.2),
            L.Flatten()
        ])

        self.conv3 = tf.keras.Sequential([
            L.Conv2D(size*16, kernel_size=(15, 15),activation='relu'),
            L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            L.Dropout(0.2),
            L.Conv2D(size*16, kernel_size=(15, 15),activation='relu'),
            L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            L.Dropout(0.2),
            L.Flatten()
        ])

        self.conv4 = tf.keras.Sequential([
            L.Conv2D(size*16, kernel_size=(20, 20),activation='relu'),
            L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            L.Dropout(0.2),
            L.Conv2D(size*16, kernel_size=(20, 20),activation='relu'),
            L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            L.Dropout(0.2),
            L.Flatten()
        ])

        self.concat  = L.Concatenate()
    
        self.dense1 = tf.keras.Sequential([
            L.Dense(size*128, activation='relu'),
            L.Dropout(0.2)
        ])

        self.dense2 = tf.keras.Sequential([
            L.Dense(size*64, activation='relu'),
            L.Dropout(0.2)
        ])

        self.model_output = L.Dense(1, activation="sigmoid")
        pass 
        
    @tf.function
    def call(self, x, training=False):

        x1 = self.conv1(x, training=training)
        x2 = self.conv2(x, training=training)
        x3 = self.conv3(x, training=training)
        x4 = self.conv4(x, training=training)

        z = self.concat([x1, x2, x3,x4])
        # z = self.flatten(z)        

        z = self.dense1(z, training=training)
        z = self.dense2(z, training=training)

        return self.model_output(z)


