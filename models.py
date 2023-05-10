from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, UpSampling2D, Input
from keras.models import Sequential, Model
import tensorflow as tf

def AutoEncoder_VGG16(input_shape):

    base = VGG16(include_top=False,
                 weights='imagenet', 
                 input_shape=input_shape)
    
    for layer in base.layers[:15]:
        layer.trainable = False
    
    output = base.output
    model = Model(inputs=base.input, outputs=output)

    return model


def Decoder(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    x=Conv2D(256, (3,3), activation='relu', padding='same')(inputs)
    x=Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x=UpSampling2D((2, 2))(x)
    x=Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x=UpSampling2D((2, 2))(x)
    x=Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x=UpSampling2D((2, 2))(x)
    x=Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x=UpSampling2D((2, 2))(x)
    x=Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    x=UpSampling2D((2, 2))(x)

    model = Model(inputs=inputs, outputs=x)
    return model
