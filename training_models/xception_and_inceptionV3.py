from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from keras.applications.xception import preprocess_input as preprocess_input_xception
import pandas as pd
from keras.preprocessing import image                  
from tqdm import tqdm
import os
import numpy as np
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.datasets import load_files   
from keras.utils import np_utils   

train_data_path = 'D:/Deep-learning/dog-breed-identification/Dataset/train_custom/'
validation_data_path = 'D:/Deep-learning/dog-breed-identification/Dataset/validation_custom/'

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 120)
    return dog_files, dog_targets

train_files, train_targets = load_dataset(train_data_path)
valid_files, valid_targets = load_dataset(validation_data_path)

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(225, 225))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def extract_Xception(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_xception(tensors)
    return Xception(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

def extract_InceptionV3(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_inception(tensors)
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

train_inceptionV3 = extract_InceptionV3(train_files)
valid_inceptionV3 = extract_InceptionV3(valid_files)
print("InceptionV3 shape", train_inceptionV3.shape[1:])

train_xception = extract_Xception(train_files)
valid_xception = extract_Xception(valid_files)
print("Xception shape", train_xception.shape[1:])

def input_branch(input_shape=None):
    
    size = int(input_shape[2] / 4)
    
    branch_input = Input(shape=input_shape)
    branch = GlobalAveragePooling2D()(branch_input)
    branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
    branch = BatchNormalization()(branch)
    branch = Activation("relu")(branch)
    return branch, branch_input

inceptionV3_branch, inceptionV3_input = input_branch(input_shape=(train_inceptionV3.shape[1:]))
xception_branch, xception_input = input_branch(input_shape=(train_xception.shape[1:]))
concatenate_branches = Concatenate()([inceptionV3_branch, xception_branch])
net = Dropout(0.3)(concatenate_branches)
net = Dense(1024, use_bias=False, kernel_initializer='uniform')(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.3)(net)
net = Dense(120, kernel_initializer='uniform', activation="softmax")(net)

model = Model(inputs=[inceptionV3_input, xception_input], outputs=[net])
model.summary()

model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])
model.fit(
    [train_inceptionV3, train_xception], train_targets, 
    validation_data=([valid_inceptionV3, valid_xception], valid_targets),
    epochs=20, 
    batch_size=4
)