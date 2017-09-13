## trains a value network. Takes the game and game result data generated from the preprocessor.

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


# Convolutional network building


BOARD = 19
PLANES = 11
LEARNING_RATE = 0.03
MODEL = 'models/value-13plane.tflearn'
REPETITIONS = 10
FILTERS= 192


network = input_data(shape=[None, BOARD, BOARD, PLANES])

network = conv_2d(network, FILTERS, 5, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = fully_connected(network, 256, activation='relu')
network = fully_connected(network, 1, activation='tanh')
network = regression(network, optimizer='SGD',
                     loss='categorical_crossentropy',
                     learning_rate=0.03)
model = tflearn.DNN(network, tensorboard_verbose=0)
#model.load('models/value-19.tflearn')

# Data loading and preprocessing

# load test data
X_test=np.load('/media/falk/6,0 TB Volume/19 full 2/test/kgsdatain_alllayer.npy')
Y_test=np.load('/media/falk/6,0 TB Volume/19 full 2/test/kgsdatawin_alllayer.npy')
X_test=X_test[:10000]
Y_test=Y_test[:10000]
X_test = np.moveaxis(X_test, 1, 3)  # exchange axis
Y_test = Y_test[:, None]


#load training data

CHUNKS = len(glob.glob("/media/falk/6,0 TB Volume/19 full/kgsdatain_alllayer_*"))

for repetitions in range(REPETITIONS):
    for i in range(int((CHUNKS/2))):  # for split datasets

        rando = list(range(CHUNKS))
        random.shuffle(rando)
        print(rando)
        filein='/media/falk/6,0 TB Volume/19 full/kgsdatain_alllayer_'+ str(rando[1])+'.npy'
        Xin1 = np.load(filein)
        filewin = '/media/falk/6,0 TB Volume/19 full/kgsdatawin_alllayer_' + str(rando[1]) + '.npy'
        Zin1 = np.load(filewin)

        filein='/media/falk/6,0 TB Volume/19 full/kgsdatain_alllayer_'+ str(rando[2])+'.npy'
        Xin2 = np.load(filein) # loads second chunk
        filewin = '/media/falk/6,0 TB Volume/19 full/kgsdatawin_alllayer_' + str(rando[2]) + '.npy'
        Zin2 = np.load(filewin)





        half1=int(round(Xin1.size/2))

        print(half1) #shuffles among chunks
        Xin = np.concatenate([Xin1[:], Xin2[:]])
        Zin = np.concatenate([Zin1[:], Zin2[:]])

        del Xin1, Xin2


        randomize = np.arange(len(Xin))
        np.random.shuffle(randomize)

        Xin = Xin[randomize]
        Zin = Zin[randomize]

        X=Xin
        Z=Zin

        # (Non-)Real-time random data augmentation before
        # loading into NN

        for l in range(1):

            randarr = np.zeros([len(X) * 3])  # random flipping and transposing 2^3
            for m in range(len(randarr)):
                randarr[m] = random.getrandbits(1)

            def flipit(a):
                if randarr[a] == 1:
                    for b in range(PLANES):
                        X[a, b] = X[a, b].transpose()
                if randarr[a + len(X)] == 1:
                    for c in range(BOARD):
                        for d in range(PLANES):
                            X[a][d][c] = X[a][d][c][::-1]
                if randarr[a + 2 * len(X)] == 1:
                    for d in range(PLANES):
                        X[a][d] = X[a][d][::-1]


            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            pool.map(flipit,range(len(X)))
            pool.close()
            pool.join()


            X = np.moveaxis(X, 1, 3) # exchange axis
            Z = Z[:, None]
            # Train using classifier
            model.fit(X, Z, n_epoch=1, shuffle=True,
            show_metric=True, batch_size=256, run_id="gocnnnn")


            model.save(MODEL)
        print("Trainingblock "+str(i + 1)+", Epoch "+str(repetitions+1)+" finished")
    print("Repetition "+str(repetitions + 1)+" finished ")
print("Training finished")


