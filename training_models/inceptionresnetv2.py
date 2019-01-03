import numpy as np
from image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import InceptionResNetV2, Xception
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt

batch_size = 64
image_size = 299

train_data_path = 'D:/Deep-learning/dog-breed-identification/Dataset/train_custom/'
validation_data_path = 'D:/Deep-learning/dog-breed-identification/Dataset/validation_custom/'

train_data_generator = ImageDataGenerator(
    rescale = 1. / 255,
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

base_model = InceptionResNetV2(
    weights = 'imagenet',
    include_top = False,
    input_shape = (299, 299, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(120, activation='softmax')(x)

model = Model(
    inputs = base_model.input,
    outputs = predictions
)

#model.summary()

# keras way to freeze layers you don't want to train
for layer in base_model.layers: layer.trainable = False

model.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit_generator(
    train_generator,
    train_generator.n // batch_size,
    epochs = 12,
    validation_data = validation_generator,
    validation_steps = validation_generator.n // batch_size
)

model.save('inceptionresnetv2_weight_no_filter.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy for inceptionResNetV2')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for inceptionResNetV2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()