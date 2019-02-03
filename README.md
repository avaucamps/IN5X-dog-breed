# Dog Breed Identification

For the kaggle competition [dog breed identification](https://www.kaggle.com/c/dog-breed-identification) 
I trained a few CNNs with Keras.

Firstly I created some custom CNNs to see how good a custom one could do, but the scores were not 
good ([custom_model.py](custom_model.py)). <br>
Then I used bottleneck features with pre-trained CNNs and I started obtaining better results 
([bottleneck_features.py](bottleneck_features.py)). <br>
Finally I fine-tuned a pre-trained model (inceptionresnetv2) and obtained even better results 
([inceptionresnetv2.py](inceptionresnetv2.py)). <br>

Then I tried improving the results by adding image processing. I applied filters to the images, and found out that the
filtering with the gamma correction slightly improved the results.

With the fine-tuned inceptionresnetv2 model I achieved a public score of 0.30041 on Kaggle.
