import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from dataset_helper import prepare_dataset
import matplotlib.pyplot as plt
from keras.applications import VGG16, VGG19

batch_size = 64
image_size = 64

prepare_dataset()
train_data_path = 'D:/Deep-learning/dog-breed-identification/Dataset/train_custom/'
validation_data_path = 'D:/Deep-learning/dog-breed-identification/Dataset/validation_custom/'
train_data_generator = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest'
)

validation_data_generator = ImageDataGenerator(
    rescale = 1. / 255
)

train_generator = train_data_generator.flow_from_directory(
    train_data_path,
    target_size = (image_size, image_size),
    batch_size = batch_size,
    class_mode = 'categorical'
)

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_path,
    shuffle = False,
    target_size = (image_size, image_size),
    batch_size = batch_size,
    class_mode = 'categorical'
)

base_model = VGG19(
    weights = 'imagenet',
    include_top = False, 
    input_shape = (64, 64, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(120, activation='softmax')(x)

model = Model(
    inputs = base_model.input,
    outputs = predictions
)

model.summary()

# keras way to freeze layers you don't want to train
for layer in base_model.layers: layer.trainable = False

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit_generator(
    train_generator,
    train_generator.n // batch_size,
    epochs = 10,
    workers = 4,
    validation_data = validation_generator,
    validation_steps = validation_generator.n // batch_size
)

