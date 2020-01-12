## Project: SpeckleDNet
## Developer: Milad Shiri
## @2020

import tools

import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

def merge_path (input_layer, filter_ratio):
    conv = layers.Conv2D(16 * filter_ratio,(3, 3), activation='relu')(input_layer)
    maxp = layers.MaxPooling2D((1, 1))(conv)
    conv = layers.Conv2D(16 * filter_ratio,(3, 3), activation='relu')(maxp)
    maxp = layers.MaxPooling2D((2, 2))(conv)
    flat = layers.Flatten()(maxp)
    dens = layers.Dense(32 * filter_ratio, activation='relu')(flat)
    merge_output = layers.Dense(1, activation='sigmoid')(dens)
    return merge_output
  

def primary_path (dims, filter_ratio):
    main_input = keras.Input(shape=dims, name='main')
    conv = layers.Conv2D(2 * filter_ratio,(3, 3), activation='relu')(main_input)
    conv = layers.Dropout(0.3)(conv)
    maxp = layers.MaxPooling2D((1, 1))(conv)
    conv = layers.Conv2D(8 * filter_ratio,(3, 3), activation='relu')(maxp)
    conv = layers.Dropout(0.3)(conv)
    maxp = layers.MaxPooling2D((2, 2))(conv)
    main_output = maxp
    return main_input, main_output


def comp_path (dims, filter_ratio):
    comp_input = keras.Input(shape=dims, name='comp')
    conv = layers.Conv2D(2 * filter_ratio,(3, 3), activation='relu')(comp_input)
    maxp = layers.MaxPooling2D((2, 2))(conv)
    conv = layers.Conv2D(4 * filter_ratio,(3, 3), activation='relu')(maxp)
    conv = layers.Dropout(0.3)(conv)
    maxp = layers.MaxPooling2D((1, 1))(conv)
    conv = layers.Conv2D(4 * filter_ratio,(3, 3), activation='relu')(maxp)
    conv = layers.Dropout(0.3)(conv)
    maxp = layers.MaxPooling2D((1, 1))(conv)
    conv = layers.Conv2D(4 * filter_ratio,(3, 3), activation='relu')(maxp)
    conv = layers.Dropout(0.3)(conv)
    maxp = layers.MaxPooling2D((1, 1))(conv)
    comp_output = maxp
    return comp_input, comp_output
    

def network(primary_path, comp_path, merge_path, main_dims, main_filter_ratio, 
                                                 comp_dims, comp_filter_ratio):
    main_input, primary_output = primary_path (main_dims, main_filter_ratio)
    comp_input, comp_output = comp_path (comp_dims, comp_filter_ratio)
    
    merge_input = layers.concatenate([primary_output, comp_output])
    
    merge_output = merge_path (merge_input, main_filter_ratio)
    model = keras.Model(inputs=[main_input, comp_input],
                        outputs=[merge_output])
    return model




main_filter_ratio = 16
comp_filter_ratio = 16

main_dims = (30, 30, 1)
comp_dims = (40, 40, 1)

data_dir = "D:/speckle_detection_dataset/"
train_xm, train_xc, train_y = pickle.load(open(
        data_dir + 'original_train_big_cleaned.pickle', 'rb'))
test_xm, test_xc, test_y = pickle.load(open(
        data_dir + 'original_test_big_cleaned.pickle', 'rb'))
validation_xm, validation_xc, validation_y = pickle.load(open(
        data_dir + 'original_validation_big_cleaned.pickle', 'rb'))

train_Xm, train_Xc, train_Y = tools.construct_input_data(
        train_xm, train_xc, train_y, main_dims, comp_dims)
test_Xm, test_Xc, test_Y = tools.construct_input_data(
        test_xm, test_xc, test_y, main_dims, comp_dims)
validation_Xm, validation_Xc, validation_Y = tools.construct_input_data(
        validation_xm, validation_xc, validation_y, main_dims, comp_dims)

model = network(primary_path, comp_path, merge_path, 
                main_dims, main_filter_ratio,
                comp_dims, comp_filter_ratio)
#    model.summary()

opt = keras.optimizers.Adam()
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy', 'binary_crossentropy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', 
                                        patience=20, restore_best_weights=True)

history = model.fit([train_Xm, train_Xc], train_Y, epochs=2, 
                    validation_data=([validation_Xm, validation_Xc], validation_Y), 
                    batch_size=100, shuffle=False, verbose=1, callbacks=[callback])

model.save('specklednet_trained_{}.h5'.format(time.strftime("%Y%m%d_%H%M%S")))
