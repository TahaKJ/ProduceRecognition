#-*- coding: utf-8 -*-
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras import optimizers
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import random
import os


# Process Model
# model = ResNet50()
#model = VGG16()
labels = ['apple1', 'apple2', 'Aprico', 'Nectarine', 'Tomato']
model = load_model('resnet50_5classes.h5') # load finetuned_model
image = load_img('target.jpg', target_size=(224, 224))  # previous image size (224, 224, 3) for resnet50
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

# Generate predictions
pred = model.predict(image)

model.summary()

print('------------------------Model : ResNet50 Finetuned----------------------------')

"""
x = decode_predictions(pred, top=1)[0]
print('Predicted:', x[0][1])

print 'Your purchase:', x[0][1]
"""
print(labels)
print(pred)

maxi=max(pred[0])
max_index = np.argmax(pred[0])

print 'Your purchase:', labels[max_index]

#  get weight  API
weight = random.randint(20,1000)
print 'The weight is :', weight,'g'

price = random.uniform(1,10) # get price from API
price = round(price) # round the price
print 'price: €',  price*weight/1000, 'per Kg: €', price

# print('Top 5 Predictions:', decode_predictions(pred, top=5)[0])
np.argmax(pred[0])

# optimizer
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
# X_test, [y_test_one, y_test_two], verbose=1
# eval = model.evaluate(image, ,  ,)
# print eval
print(model.metrics_names)


# result user imput
result = True
response = raw_input(" choose manualy? y/n :  ")

if (response == 'y'):
    result = False

else:
    result = True

label = raw_input(" enter label ==> ")



if (result):
    img = load_img('target.jpg', target_size=(224, 224))
    # saving images to build a new datasets

    # verifing if the name is used
    path = './correct'
    files = os.listdir(path)
    n=1
    for name in files:
        if label+'.jpg' == name:
            label = label + str(n)
            n=n+1

    saving_directory = 'correct/'+label+'.jpg'
    img.save(saving_directory, 'JPEG')
else:
    img = load_img('target.jpg', target_size=(224, 224))
    # saving images to build a new datasets
    # verifing if the name is used
    path = './learning'
    files = os.listdir(path)
    n=1
    for name in files:
        if label+'.jpg' == name:
            label = label + str(n)
            n=n+1
    saving_directory = 'learning/'+label+'.jpg'
    img.save(saving_directory, 'JPEG')
