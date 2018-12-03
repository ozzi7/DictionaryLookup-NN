# dictionary lookup test input [x1,y1], [x2,y2],..[q1] -> output y where q == x
# leaving the RELU layer trainable allows x and y to be outside [0,1], but values must be > 0

import numpy as np
import keras.backend as K
from keras.layers import *
from keras.initializers import *
from keras.models import *
from keras.optimizers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.generic_utils import get_custom_objects
import random
import math


def gaussian(x):
    return K.exp(-K.pow((x*999999999),2))


get_custom_objects().update({'gaussian': Activation(gaussian)})


# create samples
def create_data(nof_samples, nof_dict_entries, min_val=0, max_val=1):

    data_X = np.random.rand(nof_samples * (nof_dict_entries * 3))*(max_val-min_val)+min_val
    data_y = np.random.rand(nof_samples) * (nof_dict_entries)*(max_val-min_val)+min_val

    for i in range(0, nof_samples):
        index = random.randint(0,nof_dict_entries) # no key in dict output 0
        if index < nof_dict_entries:
            for j in range(0, nof_dict_entries):
                data_X[i*(3*nof_dict_entries) + j*3 + 2] = data_X[i*(3*nof_dict_entries) + index*3]
            data_y[i] = data_X[i*(3*nof_dict_entries) + index*3 + 1]
        else:
            key = random.uniform(min_val, max_val)
            for j in range(0, nof_dict_entries):
                data_X[i*(3*nof_dict_entries) + j * 3 + 2] = key
            data_y[i] = 0.0

    return np.reshape(data_X,[nof_samples,(nof_dict_entries)*3]), np.reshape(data_y,[nof_samples,1])


nof_samples = 50
nof_dict_entries = 4 # let this be a multiple of 2 for now
min_val = 0
max_val = 20

train_X, train_y = create_data(nof_samples, nof_dict_entries, min_val, max_val)

input = Input(shape=(nof_dict_entries*3,))

print(input)
outputs = []
# Create a larger dictionary out of the small ones
for i in range(nof_dict_entries):
    first_layer = []
    out = Lambda(lambda x: x[:, i*3:i*3+3])(input)  # get 3 values of input

    first_layer.append(Dense(1, activation="linear",
                             weights=[np.array([[0], [1], [0]]), np.array([0])],
                             trainable=False)(out)) # to forward values
    first_layer.append(Dense(1, activation="gaussian",
                             weights=[np.array([[10],[0],[-10]]), np.array([0])],
                             trainable=False)(out))  # output high value for match, 0 otherwise
    concatenate_l = concatenate(first_layer)
    outputs.append(Dense(1, activation='relu',
                         weights= [np.array([[1],[20]]), np.array([-20])],
                         trainable=True)(concatenate_l))  # subtract high value to cut off non-matching

if nof_dict_entries > 1:
    concatenate_outputs_l = concatenate(outputs)
    output_l = Dense(1, activation='linear', weights=[np.ones([nof_dict_entries,1]), np.zeros(1)],
                     trainable=False)(concatenate_outputs_l)  # sum up the two elements, identity
else:
    output_l = Dense(1, activation='linear', weights=[np.array([[1]]), np.array([0])], trainable=False)(outputs[0])

model = Model(inputs=[input], outputs=[output_l])
model.compile(optimizer=Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mae')
model.summary()
model.fit(train_X, train_y, epochs=1000, batch_size=1000)

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())
print(model.layers[3].get_weights())
print(model.layers[4].get_weights())
print(model.layers[9].get_weights())
print(model.layers[10].get_weights())

test_X, test_y = create_data(200, nof_dict_entries, min_val, max_val)
results = model.predict(test_X, batch_size=200)

error = []
error_baseline = []
for i in range(0, len(test_y)):
    print('Input:', test_X[i], 'Result:', test_y[i], "Prediction:", results[i], 'Error:',\
         '%.2f' %(abs(test_y[i] - results[i])))
    error.append(abs(test_y[i] - results[i]))
    average = np.sum(test_X[i,1::2])/nof_dict_entries
    error_baseline.append(abs(average - test_y[i]))

print('\n MAE when average of values predicted:', np.sum(np.array(error_baseline))/len(error_baseline))
print('\n MAE for 2 entry dictionary when predicting average: 1/6*(max_value-min_value)', (1/6)*(max_val-min_val))
print('\n MAE NN:', np.sum(np.array(error))/len(error))


