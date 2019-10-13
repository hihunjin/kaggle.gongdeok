import tensorflow as tf 
import tensorflow.keras.layers as L

def save(model, path):
    model_json = model.to_json()
    with open("model.json", "w") as json_file : 
        json_file.write(model_json)

class simpleConvNet(tf.keras.Model):
    
    def __init__(self, size=4,**kwargs):
        super(simpleConvNet, self).__init__(**kwargs)

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
    def call(self, x, training=False):

        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)

        x = self.flatten(x)        

        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)

        return self.model_output(x)