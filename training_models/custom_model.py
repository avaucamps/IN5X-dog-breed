import numpy as numpy
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from dataset_helper import prepare_dataset
from data_generator import get_train_generator, get_validation_generator

batch_size = 64
image_size = (64, 64)

prepare_dataset()

train_generator = get_train_generator(batch_size, image_size)
validation_generator = get_validation_generator(batch_size, image_size)

# Create model
model = Sequential()

# Layer 1
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (image_size, image_size, 3)))
model.add(MaxPooling2D(pool_size = 2))

# Layer 2
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))

# Layer 3
model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))

# Layer 4
model.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))

# Fully connected layers
model.add(Flatten(name = 'flatten'))
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))

# Output
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