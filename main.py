# dictionary lookup test input [x1,y1,q1], [x2,y2,q1],.. -> output y where q == x
# outputs 0 if key not found

import numpy as np
import keras.backend as K
from keras.layers import *
from keras.initializers import *
from keras.models import *
from tensorflow.keras.optimizers import *
from keras.utils.generic_utils import get_custom_objects
import random


# basically inf between -1 and 1, outside 0, and 1 at -1 and 1
def gaussian(x):
    return K.exp(-K.pow((x*100000),2))

get_custom_objects().update({'gaussian': Activation(gaussian)})


# create samples
def create_data(nof_samples, nof_dict_entries, min_val=0, max_val=1):

    data_X = np.random.rand(nof_samples * (nof_dict_entries * 3))*(max_val-min_val)+min_val
    data_y = np.random.rand(nof_samples) * nof_dict_entries * (max_val - min_val) + min_val

    for i in range(0, nof_samples):
        index = random.randint(0,nof_dict_entries) # no key in dict output 0
        if index < nof_dict_entries:
            for j in range(0, nof_dict_entries):
                # sets all second elements to the same as the first one of key
                data_X[i*(3*nof_dict_entries) + j*3 + 2] = data_X[i*(3*nof_dict_entries) + index*3]
            data_y[i] = data_X[i*(3*nof_dict_entries) + index*3 + 1]
        else:
            key = random.uniform(min_val, max_val)
            for j in range(0, nof_dict_entries):
                data_X[i*(3*nof_dict_entries) + j * 3 + 2] = key
            data_y[i] = 0.0

    return np.reshape(data_X,[nof_samples,(nof_dict_entries)*3]), np.reshape(data_y,[nof_samples,1])


nof_samples = 500000
nof_dict_entries = 16 # let this be a multiple of 2 for now
min_val = 0
max_val = 20

train_X, train_y = create_data(nof_samples, nof_dict_entries, min_val, max_val)

input = Input(shape=(nof_dict_entries*3,))

outputs = []
# Create a larger dictionary out of the small ones
for i in range(nof_dict_entries):
    first_layer = []

    F = Lambda(lambda x, start, end: x[:,start:end], arguments = {'start': i*3, 'end':i*3+3})(input)

    first_layer.append(Dense(1, activation="linear",
                                weights=[np.array([[0], [1], [0]]), np.array([0])],
                                trainable=False)(F)) # to forward values
    first_layer.append(Dense(1, activation="gaussian",
                                weights=[np.array([[-1],[0],[1]]), np.array([0])],
                                trainable=False)(F))  # output 1 for match, 0 otherwise
    concatenate_l = concatenate(first_layer)
    outputs.append(Dense(1, activation='relu',
                            weights= [np.array([[1],[max_val]]), np.array([-max_val])],
                            trainable=False)(concatenate_l))  # subtract high value to cut off non-matching

if nof_dict_entries > 1:
    concatenate_outputs_l = concatenate(outputs)
    output_l = Dense(1, activation='linear', weights=[np.ones([nof_dict_entries,1]), np.array([0])],
                     trainable=False)(concatenate_outputs_l)  # sum up all outputs
else:
    output_l = Dense(1, activation='linear', weights=[np.array([[1]]), np.array([0])], trainable=False)(outputs[0])

model = Model(inputs=[input], outputs=[output_l])
model.compile(optimizer=Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mae')
model.summary()
model.fit(train_X, train_y, epochs=50, batch_size=1000000)

test_X, test_y = create_data(3, nof_dict_entries, min_val, max_val)
results = model.predict(test_X, batch_size=1)

error = []
error_baseline = []
for i in range(0, len(test_y)):
    print('Input:', test_X[i], 'Result:', test_y[i], "Prediction:", results[i], 'Error:',\
         '%.2f' %(abs(test_y[i] - results[i])))
    error.append(abs(test_y[i] - results[i]))
    average = np.sum(test_X[i,1::2])/nof_dict_entries
    error_baseline.append(abs(average - test_y[i]))

print('\n MAE NN:', np.sum(np.array(error))/len(error))