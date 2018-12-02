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
    return K.exp(-K.pow((x*100),2))


get_custom_objects().update({'gaussian': Activation(gaussian)})


# create samples
def create_data(nof_samples, nof_dict_entries, min_val=0, max_val=1):

    data_X = np.random.rand(nof_samples * (nof_dict_entries*2+1))*(max_val-min_val)+min_val
    data_y = np.random.rand(nof_samples) * (max_val-min_val)+min_val
    for i in range(0, nof_samples):
        index = random.randint(0,nof_dict_entries) # 1/3rd chance of no key in dict, we need it to output 0 here
        if index < nof_dict_entries:
            data_X[i*(nof_dict_entries*2+1) + nof_dict_entries*2] = data_X[i*(nof_dict_entries*2+1) + index*2]
            data_y[i] = data_X[i*(nof_dict_entries*2+1) + index*2 + 1]
        else:
            data_X[i * (nof_dict_entries * 2 + 1) + nof_dict_entries * 2] = random.uniform(min_val, max_val)
            data_y[i] = 0.0

    return np.reshape(data_X,[nof_samples,nof_dict_entries*2+1]), np.reshape(data_y,[nof_samples,1])


nof_samples = 1000
nof_dict_entries = 2
min_val = 0
max_val = 20

train_X, train_y = create_data(nof_samples, nof_dict_entries, min_val, max_val)

input = Input(shape=(nof_dict_entries*2+1,))
#x = Dropout(0.2)(input)
branch_linear_l = Dense(2, activation="linear")(input)  # to forward values
branch_gaussian_l = Dense(2, activation="gaussian")(input)  # output high value for match, 0 otherwise
concatenate_l = concatenate([branch_linear_l, branch_gaussian_l])
relu_l = Dense(2, activation='relu')(concatenate_l)  # subtract high value to cut off non-matching
output_l = Dense(1, activation='linear')(relu_l)  # sum up the two elements, identity

model = Model(inputs=[input], outputs=[output_l])

weights = model.layers[2].get_weights()

branch_linear_w = [np.array([[0,0],[1,0],[0,0],[0,1],[0,0]]), np.array([0,0])]
branch_gaussian_w = [np.array([[2,0],[0,0],[0,2],[0,0],[-2,-2]]), np.array([0,0])]
#relu_w = [np.array([[1,0],[0,1],[(max_val-min_val),0],
#                           [0,(max_val-min_val)]]), np.array([-(max_val-min_val),-(max_val-min_val)])]
#relu_w = [np.array([[1,0],[0,1],[1,0], [0,1]]), np.array([-1,-1])] # leaving this for training works
relu_w = [np.array([[1,0],[0,1],[4.5,-7.0], [-5.5,3]]), np.array([-4.5,-3])] # alternative

output_w = [np.array([[1],[1]]), np.array([0])]

model.layers[1].set_weights(branch_linear_w)
model.layers[2].set_weights(branch_gaussian_w)
model.layers[4].set_weights(relu_w)
model.layers[5].set_weights(output_w)

model.layers[1].trainable = False # this is just a passthrough layer
model.layers[2].trainable = False # nothing to be gained, same for all dictionaries
model.layers[4].trainable = True # allows the key and value to be from outside [0,1]
model.layers[5].trainable = False # sums up the values, same for all dictionaries

model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mae')
model.summary()
model.fit(train_X, train_y, epochs=40000, batch_size=1000)

print(model.layers[1].get_weights())
print(model.layers[2].get_weights())
print(model.layers[4].get_weights())
print(model.layers[5].get_weights())

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


