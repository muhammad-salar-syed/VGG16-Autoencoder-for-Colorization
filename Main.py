
from keras.layers import Conv2D, UpSampling2D, Input
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, gray2rgb
import numpy as np
import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import glob
import cv2
from skimage.transform import resize
from models import AutoEncoder_VGG16, Decoder


input_shape=(224,224,3)

model=AutoEncoder_VGG16(input_shape)
model.summary()

img_list=glob.glob('./Data/train/*.*')

X =[]
Y =[]
for i in img_list:
    img=cv2.imread(i)/255.
    r_img=cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
    lab = rgb2lab(r_img)
    X.append(lab[:,:,0]/100) 
    Y.append(lab[:,:,1:] / 128)
    
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,))
print(X.shape)
print(Y.shape)

extracted_features = []
for i, img in enumerate(X):
  I = gray2rgb(img)
  I = I.reshape((1,224,224,3))
  prediction = model.predict(I)
  prediction = prediction.reshape((7,7,512))
  extracted_features.append(prediction)
  
extracted_features = np.array(extracted_features)
print(extracted_features.shape)

decoder=Decoder((7,7,512))
decoder.summary()

decoder.compile(optimizer='Adam', loss='mse' , metrics=['accuracy'])
history=decoder.fit(extracted_features, Y, verbose=1, epochs=100, batch_size=32)

decoder.save('./Autoencoder_VGG16.hdf5')

