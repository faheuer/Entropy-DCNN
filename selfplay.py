
## This script trains the RL policy network.
# With a given policy network, a game will be played against itself and stored.
# A value network scores the game. Then moves are selected for trainig.

from __future__ import division, print_function, absolute_import

import numpy as np
import random
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.estimator import regression
import features

BOARD = 19
PLANES = 13
LEARNING_RATE = 0.03
FILTERS= 192
MODEL = 'models/policy_kgs.tflearn'
VALUEMODEL='models/value_kgs.tflearn'


# Create network
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

# Load model, network must be equivalent to model

model = tflearn.DNN(network, tensorboard_verbose=0)
model.load(MODEL, weights_only=True)


network2 = input_data(shape=[None, PLANES, BOARD, BOARD])
network2 = conv_2d(network2, FILTERS, 5, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = conv_2d(network2, FILTERS, 3, activation='relu')
network2 = fully_connected(network2, 1, activation='tanh') #remeber tanh!
network2 = regression(network2, optimizer='SGD',
                     loss='categorical_crossentropy',
                     learning_rate=0.03)

valuemodel = tflearn.DNN(network2, tensorboard_verbose=0)
#valuemodel.load(VALUEMODEL, weights_only=True)


stackwin = np.array([], dtype=int)
stackpicks = np.array([], dtype=int)
winpicks = np.array([], dtype=int)

alphabet = ["A","B","C","D","E","F","G","H","J","K","L","M","N","O","P","Q","R","S","T"]

def prediction(gamemove, col):
    predictmove=gamemove[None,:]
    resultmatrix = model.predict(predictmove)
    resultmatrix = np.reshape(resultmatrix, [BOARD,BOARD])
    result = np.unravel_index(resultmatrix.argmax(), resultmatrix.shape)
    while gamemove[4][result[0]][result[1]]==0: # if move is illegal
        resultmatrix[result[0]][result[1]] = 0 # delete from softmax
        result = np.unravel_index(resultmatrix.argmax(), resultmatrix.shape) #find second best option
    resultmatrix = np.round_(resultmatrix, 1)*10
    print(alphabet[result[1]-BOARD], BOARD-result[0])
    tmp = gamemove[0].copy()
    gamemove[0]=gamemove[1]
    gamemove[1]=tmp
    gamemove = features.extract(gamemove, result[0], result[1], col)
    return gamemove, resultmatrix



def printgame(final, resultmatrix):
    printfinal=final[0]+2*final[1]
    printfinal2=np.chararray((BOARD, BOARD))
    printfinal2[:]="."
    printfinal2[(printfinal==1)]="X"
    printfinal2[printfinal==2]="O"
    resultmatrix2=np.chararray((BOARD, BOARD))
    resultmatrix2[:]=resultmatrix[:]
    resultmatrix2[(resultmatrix2=="0")]="."
    printfinal3=np.chararray((BOARD+1,2*BOARD+3))
    for i in range(BOARD):
        printfinal3[i][1:(BOARD+1)] = printfinal2[i][0:BOARD]
        printfinal3[i][(BOARD+3):(2*BOARD+3)] = resultmatrix2[i][0:BOARD]
        printfinal3[BOARD][i+1]=alphabet[i]
        printfinal3[BOARD][i+3+BOARD]=alphabet[i]
        printfinal3[i][0]=(BOARD-i)%10
        printfinal3[i][BOARD+2]=(BOARD-i)%10
        printfinal3[i][BOARD+1]=" "
    printfinal3[BOARD][0]="."
    printfinal3[BOARD][BOARD+2]="."
    for i in range(BOARD+1):
        print(" ".join(printfinal3[i]))

def game_ini(): #0:white 1: black 2: 2 Handi etc.
    game = np.zeros((PLANES, BOARD, BOARD), dtype=int)
    game[2:5] = np.ones([3, BOARD, BOARD])  # free plane: initially all free. And a plane filled with ones
    stackB = np.array([], dtype=int)
    stackW = np.array([], dtype=int)
    i=0


    while 1 in game[4]: # while there is still a legal move
    #while i < 200:
        if i%2==0: col="W" # =b to play
        else: col="B"
        i += 1
        print(col)
        game, resultmatrix=prediction(game, col)
        printgame(game, resultmatrix)
        storegame = game[None, :]
        if col=="W":
            stackB = np.concatenate([stackB, storegame]) if stackB.size else storegame
        if col=="B":
            stackW = np.concatenate([stackW, storegame]) if stackW.size else storegame

    valueprediction=game[None,:]
    valuemodel.load(VALUEMODEL, weights_only=True)
    valueresult = valuemodel.predict(valueprediction)
    print(valueresult)
    global stackwin, stackpicks, winpicks
    if valueresult <= 0.5: #valuemodel predicts white win
        stackwin = np.concatenate([stackwin, stackW]) if stackwin.size else stackW
        winpicks = np.concatenate([winpicks, [1,1,1,0,0,0]]) if winpicks.size else [1,1,1,0,0,0]
    else:
        stackwin = np.concatenate([stackwin, stackB]) if stackwin.size else stackB
        winpicks = np.concatenate([winpicks, [0,0,0,1,1,1]]) if winpicks.size else [0,0,0,1,1,1]


    a=stackW[random.randint(0, stackW.size)]
    b=stackW[random.randint(0, stackW.size)]
    c=stackW[random.randint(0, stackW.size)]
    d=stackB[random.randint(0, stackB.size)]
    e=stackB[random.randint(0, stackB.size)]
    f=stackB[random.randint(0, stackB.size)]

    picks = np.concatenate([a,b,c,d,e,f])
    stackpicks = np.concatenate([stackpicks, picks]) if stackpicks.size else picks





game_ini() # iniciates game with given HC, 0w 1b
