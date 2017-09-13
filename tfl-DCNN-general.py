## trains a policy network. Board size, planes, learning rate, filter size and model path can be edited. It will load
# the numpy tensors generated from gameprep_general and train them. During that, it randomly shuffles them
# and flips/transposes. To increase the distribution of positions, they are stored after randomization.
# "Repetitions" denote roughly epochs.

from __future__ import division, print_function, absolute_import

import numpy as np
import glob
import tflearn
import random
import multiprocessing
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.estimator import regression


BOARD = 19
PLANES = 13
LEARNING_RATE = 0.03
MODEL = 'models/policy_new.tflearn'
REPETITIONS = 10
FILTERS= 192
RUN_ID= 'go_cnn_baduk2'


# Convolutional network building

network = input_data(shape=[None, PLANES, BOARD, BOARD])

network = conv_2d(network, FILTERS, 5, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = conv_2d(network, FILTERS, 3, activation='relu')
network = fully_connected(network, BOARD * BOARD, activation='softmax')
network = regression(network, optimizer='SGD',
                     loss='categorical_crossentropy',
                     learning_rate=LEARNING_RATE)
model = tflearn.DNN(network, tensorboard_verbose=0)
#model.load(MODEL) # use when retraining a model

# load test set

X_test=np.load("/media/falk/6,0 TB Volume/19 badukmovies/kgsdatain_alllayer.npy")
Y_test=np.load("/media/falk/6,0 TB Volume/19 badukmovies/kgsdataout_alllayer.npy")
Y_test = Y_test.reshape(len(Y_test), (BOARD * BOARD))
X_test=X_test[10000:20000]
Y_test=Y_test[10000:20000]
print(len(X_test))

# Data loading and preprocessing. Data is chopped into chunks.

CHUNKS = len(glob.glob("/media/falk/6,0 TB Volume/19 badukmovies/pro2/kgsdatain_alllayer_*"))

for repetitions in range(REPETITIONS):
    for i in range(int((CHUNKS/2))):  # for split datasets

        rando = list(range(CHUNKS))
        random.shuffle(rando)
        print(rando)
        filein = '/media/falk/6,0 TB Volume/19 badukmovies/pro2/kgsdatain_alllayer_' + str(rando[1]) + '.npy'
        Xin1 = np.load(filein)
        fileout = '/media/falk/6,0 TB Volume/19 badukmovies/pro2/kgsdataout_alllayer_' + str(rando[1]) + '.npy'
        Yin1 = np.load(fileout)
        filewin = '/media/falk/6,0 TB Volume/19 badukmovies/pro2/kgsdatawin_alllayer_' + str(rando[1]) + '.npy'
        Zin1 = np.load(filewin)
	
	#loads 2 chunks
        filein2 = '/media/falk/6,0 TB Volume/19 badukmovies/pro2/kgsdatain_alllayer_' + str(rando[2]) + '.npy'
        Xin2 = np.load(filein2)
        fileout2 = '/media/falk/6,0 TB Volume/19 badukmovies/pro2/kgsdataout_alllayer_' + str(rando[2]) + '.npy'
        Yin2 = np.load(fileout2)
        filewin2 = '/media/falk/6,0 TB Volume/19 badukmovies/pro2/kgsdatawin_alllayer_' + str(rando[2]) + '.npy'
        Zin2 = np.load(filewin2)



        half1=int(round(Xin1.size/2))
        half2=int(round(Xin2.size/2))
        print(half1)

	#shuffles chunks
        Xin3 = np.concatenate([Xin1[:half1], Xin2[half2:]])
        Xin4 = np.concatenate([Xin1[half1:], Xin2[:half2]])
        Yin3 = np.concatenate([Yin1[:half1], Yin2[half2:]])
        Yin4 = np.concatenate([Yin1[half1:], Yin2[:half2]])
        Zin3 = np.concatenate([Zin1[:half1], Zin2[half2:]])
        Zin4 = np.concatenate([Zin1[half1:], Zin2[:half2]])


        randomize = np.arange(len(Xin3))
        np.random.shuffle(randomize)

        Xin3 = Xin3[randomize]
        Yin3 = Yin3[randomize]
        Zin3 = Zin3[randomize]

        randomize = np.arange(len(Xin4))
        np.random.shuffle(randomize)

        Xin4 = Xin4[randomize]
        Yin4 = Yin4[randomize]
        Zin4 = Zin4[randomize]

	#stores shuffled chunks
        np.save(filein, Xin3)
        np.save(fileout, Yin3)
        np.save(filein2, Xin4)
        np.save(fileout2, Yin4)
        np.save(filewin, Zin3)
        np.save(filewin2, Zin4)

        for l in range(2):
            if l==0:
                X=Xin3
                Y=Yin3
            if l==1:
                X=Xin4
                Y=Yin4

        # (Non-)Real-time random data augmentation before
        # loading into NN


            randarr = np.zeros([len(X) * 3])  # random flipping and transposing 2^3
            for m in range(len(randarr)):
                randarr[m] = random.getrandbits(1)

            def flipit(a):
                if randarr[a] == 1:
                    Y[a] = Y[a].transpose()
                    for b in range(PLANES):
                        X[a, b] = X[a, b].transpose()
                if randarr[a + len(X)] == 1:
                    for c in range(BOARD):
                        Y[a][c] = Y[a][c][::-1]
                        for d in range(PLANES):
                            X[a][d][c] = X[a][d][c][::-1]
                if randarr[a + 2 * len(X)] == 1:
                    Y[a] = Y[a][::-1]
                    for d in range(PLANES):
                        X[a][d] = X[a][d][::-1]


            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            pool.map(flipit,range(len(X)))
            pool.close()
            pool.join()





            # Training and testing split and shuffle

            Y = Y.reshape(len(Y), (BOARD * BOARD))


            # Train using classifier
            model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=(X_test, Y_test),
            show_metric=True, batch_size=256, run_id=RUN_ID)


            model.save(MODEL)
        print("Trainingblock "+str(i + 1)+", Epoch "+str(repetitions+1)+" finished")
    print("Repetition "+str(repetitions + 1)+" finished ")
print("Training finished")


