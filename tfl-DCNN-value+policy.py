# the meta network. 7-1 split.
# Trains a network on professional moves together with the game outcome. Data is taken from the
# preprocessor. It can be modified by shifting layers around in the network

from __future__ import division, print_function, absolute_import

import numpy as np
import glob
import tflearn
import random
import multiprocessing
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge


# Convolutional network building

network = input_data(shape=[None, 19, 19, 2])

network = conv_2d(network, 64, 5, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')

network2 = conv_2d(network, 128, 3, activation='relu')

network = conv_2d(network, 128, 3, activation='relu')

network = fully_connected(network, 19 * 19, activation='softmax')
network2 = fully_connected(network2, 2, activation='softmax')
network = regression(network, optimizer='SGD',
                     loss='categorical_crossentropy',
                     learning_rate=0.03)
network2 = regression(network2, optimizer='SGD',
                     loss='categorical_crossentropy',
                     learning_rate=0.03)
network = merge([network,network2], mode='concat')
policyvalue = tflearn.DNN(network, tensorboard_verbose=0)

# Data loading and preprocessing

# load test data
X_test=np.load('/media/falk/268f4ebf-b908-4169-9db8-13da21e9cbec/19 prepdata/kgsdatain19.npy')
Y_test=np.load('/media/falk/268f4ebf-b908-4169-9db8-13da21e9cbec/19 prepdata/kgsdataout19.npy')
Z_test=np.load('/media/falk/268f4ebf-b908-4169-9db8-13da21e9cbec/19 prepdata/kgsdatawin19.npy')
X_test=X_test[:8000]
Y_test=Y_test[:8000]
Z_test=Z_test[:8000]
X_test = np.moveaxis(X_test, 1, 3)  # exchange axis
Y_test = Y_test.reshape(len(Y_test), (19 * 19))


# load training data
for repetitions in range(16):
    for i in range(len(glob.glob('/media/falk/268f4ebf-b908-4169-9db8-13da21e9cbec/19 prepdata/u/kgsdatain19_*'))):  # for split datasets. Source pathes can be modified here
        filein='/media/falk/268f4ebf-b908-4169-9db8-13da21e9cbec/19 prepdata/u/kgsdatain19_'+ str(i)+'.npy'
        Xin=np.load(filein)
        fileout='/media/falk/268f4ebf-b908-4169-9db8-13da21e9cbec/19 prepdata/u/kgsdataout19_'+ str(i)+'.npy'
        Yin=np.load(fileout)
        filewin='/media/falk/268f4ebf-b908-4169-9db8-13da21e9cbec/19 prepdata/u/kgsdatawin19_'+ str(i)+'.npy'
        Yin2=np.load(filewin)




        # (Non-)Real-time random data augmentation before
        # loading into NN

        flips = repetitions
        if 0 == 0:

            X, Y, Z = Xin, Yin, Yin2

            def flipit(a):
                if flips%2 == 0:
                    Y[a] = Y[a].transpose()
                    for b in range(2):
                        X[a, b] = X[a, b].transpose()
                if (round(flips/2))%2 == 0:
                    for c in range(19):
                        Y[a][c] = Y[a][c][::-1]
                        for d in range(2):
                            X[a][d][c] = X[a][d][c][::-1]
                if (round(flips/4))%2:
                    Y[a] = Y[a][::-1]
                    for d in range(2):
                        X[a][d] = X[a][d][::-1]
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            pool.map(flipit,range(len(X)))
            pool.close()
            pool.join()

            X=np.moveaxis(X,1,3) #exchange axis
            Y = Y.reshape(len(Y), (19 * 19))


            # Train using classifier
            policyvalue.fit(X, [Y, Z], n_epoch=1, shuffle=True, validation_set=(X_test, [Y_test, Z_test]),
            show_metric=True, batch_size=256, run_id='go_cnn')


            policyvalue.save('models/policyvalue-new.tflearn')
            print("Trainingblock "+str(i)+" in flip "+str(flips)+" finished")
print("Training finished")


