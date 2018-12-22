import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from dataset_helper import prepare_dataset

batch_size = 64
image_size = 224

prepare_dataset()
train_data_path = 'D:/Deep-learning/dog-breed-identification/Dataset/train_custom/'
validation_data_path = 'D:/Deep-learning/dog-breed-identification/Dataset/validation_custom/'

train_data_generator = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
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

model = Sequential()

model.add(Conv2D(32, (11, 11), activation='relu', input_shape = (image_size, image_size, 3)))
model.add(MaxPooling2D((3, 3), strides = (2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu', input_shape = (image_size, image_size, 3)))
model.add(MaxPooling2D((3, 3), strides = (2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', input_shape = (image_size, image_size, 3)))
model.add(MaxPooling2D((3, 3), strides = (2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', input_shape = (image_size, image_size, 3)))
model.add(MaxPooling2D((3, 3), strides = (2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', input_shape = (image_size, image_size, 3)))
model.add(MaxPooling2D((3, 3), strides = (2, 2)))

model.add(Flatten(name = 'flatten'))
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(120, activation = 'softmax'))

model.summary()

model.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit_generator(
    train_generator,
    train_generator.n // batch_size,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = validation_generator.n // batch_size
)