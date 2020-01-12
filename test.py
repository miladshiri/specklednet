## Project: SpeckleDNet
## Developer: Milad Shiri
## @2020

import tools
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


data_dir = "speckle_detection_dataset/"
sepckleDNet_file = 'specklednet_trained.h5'

## Loading the raw data including pathes with different dimensions
train_xm, train_xc, train_y = pickle.load(open(
        data_dir + 'train.pickle', 'rb'))
test_xm, test_xc, test_y = pickle.load(open(
        data_dir + 'test.pickle', 'rb'))
validation_xm, validation_xc, validation_y = pickle.load(open(
        data_dir + 'validation.pickle', 'rb'))

## Resizing the data to fit SpeckleDNet
main_dims = (30, 30)
comp_dims = (40, 40)

train_Xm, train_Xc, train_Y = tools.construct_input_data(train_xm, train_xc, train_y,
                                       main_dims, comp_dims)
test_Xm, test_Xc, test_Y = tools.construct_input_data(test_xm, test_xc, test_y,
                                       main_dims, comp_dims)
validation_Xm, validation_Xc, validation_Y = tools.construct_input_data(validation_xm, validation_xc, validation_y,
                                       main_dims, comp_dims)

## Loading the pre-trained SpeckleDNet
SpeckleDNet = tf.keras.models.load_model(sepckleDNet_file)


## Using SpeckleDNet for labeling the test data
y_pred = SpeckleDNet.predict([test_Xm, test_Xc])


## Calculating the statistics
y_threshold = np.zeros_like(y_pred) 
y_threshold[y_pred > .5] = 1
PRFS = precision_recall_fscore_support(test_Y, y_threshold)
precision = PRFS[0][0]
recall = PRFS[1][0]
F1_score = 2*((precision*recall)/(precision+recall))
acc = accuracy_score(test_Y, y_threshold)

results = "Precision = {}, Recall = {}, F1-score = {}, ACC = {}".format(precision,
                       recall, F1_score, acc)
print (results)


