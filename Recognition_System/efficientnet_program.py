# import all required libraries
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
# this will add progress bar to "for loop"

# this will list all files in dataset
files=os.listdir("/content/drive/MyDrive/Face_recognition/dataset")
files

path = "/content/drive/MyDrive/Face_recognition/dataset/"

# now we will prepare our image array and label array
image_array=[]
label_array=[] # it's list later I will convert it into array

# loop through each dataset
for i in range(len(files)):
    # list of all files in each face folder
    # path+files[i] -> complete path
    file_sub=os.listdir(path+files[i])
    for k in tqdm(range(len(file_sub))):
        # loop through each faces
        try:
            img=cv2.imread(path+files[i]+"/"+file_sub[k])
            # path+files[i]+"/"+file_sub[k] will look like this
            #../input/face-recognition-30/dataset/David_Schwimmer/1.jpg

            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
             # convert image color from BGR to RGB
            # resize the image to (96,96)
            img=cv2.resize(img,(96,96))
            # if you want to train on different size change this
            # training on small size like (48,48) will increase frame rate in app
            # training on large size like (192,192) will increase accuracy but will
            # decrease frame rate in app
            image_array.append(img)
            # add it to image_array
            label_array.append(i)
            # i is changing from 0-29
            # so we will use it as our label
        except:
            pass
        
# now we will clear our ram memory
import gc
gc.collect()

# now we will convert image_array list to array
# we also divide image by 255 to scale image from 0-255 to 0-1
image_array=np.array(image_array)/255.0
label_array=np.array(label_array)

# now we will split and shuffle our dataset fro training and validation
from sklearn.model_selection import train_test_split
#                                              images      label             spliting ratio
X_train,X_test,Y_train,Y_test=train_test_split(image_array,label_array,test_size=0.15)

from keras import layers,callbacks,utils, applications,optimizers
from keras.models import Sequential, Model, load_model

import tensorflow as tf

from tensorflow.keras import Sequential, layers
import tensorflow as tf

# Create model
model = Sequential()

# EfficientNetB0 pre-trained model with input shape (96, 96, 3)
pretrained_model = tf.keras.applications.EfficientNetB0(input_shape=(96, 96, 3),
                                                        include_top=False,
                                                        weights="imagenet")

# Add it to our Sequential model
model.add(pretrained_model)

# Before adding GlobalAveragePooling2D, add a layer to define the output shape
#model.add(layers.Reshape( (3, 3, 1280)))

# GlobalAveragePooling2D without reshaping
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))

# Add dense layer with number of output
model.add(layers.Dense(1))

# Show model summary
model.summary()

# now compile model
# small error
model.compile(optimizer="adam",loss="mean_squared_error",metrics=["mae"])
# optimizer: You can change it to some other optimizer to improve accuracy(SGD,etc)
# loss: Try different losses to improve accuracy

# create a checkpoint to save best accuracy model
ckp_path="trained_model/model"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                                   monitor="val_mae",
                                                   mode="auto",
                                                   save_best_only=True,
                                                   save_weights_only=True)
# monitor="val_mae": watch mae of validation set and when it decrease save the model
#mode: it is use to check for decrease or increase in mae
# mode:(auto, min,max)
# save_weights_only: if true save only weight( matrix of number)


# create a lr reducer which will decrease learning rate when accuracy does't increase
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(factor=0.9,
                                             monitor="val_mae",
                                             mode="auto",
                                             cooldown=0,
                                             patience=5,
                                             verbose=1,
                                             min_lr=1e-6)
#patience: wait for 5 epoch then decrease learnning rate
# verbose : show accuracy(val_mae) every epoch
#min_lr=minimum learning rate

# you can choose other reduce_lr to increase accuracy


# train model
Epoch=30
Batch_Size=64
history=model.fit(X_train,Y_train
                 ,validation_data=(X_test,Y_test),
                  batch_size=Batch_Size,
                  epochs=Epoch,
                  callbacks=[model_checkpoint,reduce_lr])


# after the training is finished
# load best model
model.load_weights(ckp_path)

# convert best model to tensorflow lite format
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()
# save model
with open("model.tflite","wb") as f:
    f.write(tflite_model)
    
    
# if you want to see prediction of validatation set
prediction_val=model.predict(X_test,batch_size=64)
prediction_val[:20]# show first 20 value


# original label
X_test[:20]
# now I will show you already trained model
# link will be in the description