from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_resnet_v2
import pandas as pd
import sys

model = load_model('C:\\Users\\Antoine\\Desktop\\IN5X-dog-breed\\inceptionresnetv2_weight.h5')

def predict_image(path, model):
    classes = np.load('C:\\Users\\Antoine\\Desktop\\IN5X-dog-breed\\classes.npy')
    img = image.load_img(path, target_size=(299, 299))
    preprocessed_image = np.empty((1,299,299,3))
    preprocessed_image[0] = inception_resnet_v2.preprocess_input(image.img_to_array(img))

    prediction = model.predict(preprocessed_image)

    indexClass = np.argmax(prediction)
    print("Le chien dans l'image est un " + classes[indexClass])

action = "o"
while (action == "o"):
    path = input("Veuillez entrer le chemin auquel se trouve l'image à prédire.\n")
    predict_image(path, model)
    action = input("Pour prédire une image, appuyez sur o. Pour quitter appuyez sur une autre touche.\n")

sys.exit()