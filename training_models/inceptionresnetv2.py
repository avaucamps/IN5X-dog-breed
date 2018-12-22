import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import InceptionV3
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from dataset_helper import prepare_dataset

batch_size = 64
image_size = 75

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
    batch_size = 64,
    class_mode = 'categorical'
)

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_path,
    shuffle = False,
    target_size = (image_size, image_size),
    batch_size = 64,
    class_mode = 'categorical'
)

base_model = InceptionV3(
    weights = 'imagenet',
    include_top = False,
    input_shape = (75, 75, 3)
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
    validation_data = validation_generator,
    validation_steps = validation_generator.n // batch_size
)