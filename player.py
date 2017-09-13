
## Game simulator for a selected policy model
# Enter the model path in "MODEL:...". Enter handicap in "HC=...". If there is no handicap, with HC=0 the bot is
# white, HC=1 the bot is black. Then run the code in python. The command: game = play("<letter-digit>", game)
# plays a move at the given coordinates and returns a computer move.

from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.estimator import regression
import features

BOARD = 19
PLANES = 13
MODEL = 'models/policy_kgs.tflearn'
HC = 1 # iniciates game with given HC, 0>w 1>b
FILTERS= 192

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
                     learning_rate=0.03)

# Load model, network must be equivalent to model
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load(MODEL)
BOT = "B"
nBOT = "W"


alphabet = ["A","B","C","D","E","F","G","H","J","K","L","M","N","O","P","Q","R","S","T"] #for decoding coordinates

def playing(colrow,g): # takes a board position and adds a players move
    col,row=colrow[0],colrow[1:3]
    row=BOARD-int(row)
    col=alphabet.index(col)
    tmp = g[0].copy()
    g[0]=g[1]
    g[1]=tmp
    g = features.extract(g, row, col, BOT)
    return prediction(g)

def prediction(gamemove): #takes a board positions and returns a bot prediction
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
    gamemove = features.extract(gamemove, result[0], result[1], nBOT)
    return gamemove, resultmatrix


def play(field,r): # wraps the printfunction
    final, resultmatrix=playing(field,r.copy())
    printgame(final, resultmatrix)
    return final

def printgame(final, resultmatrix): # prints the game formatted and the probability matrix aside
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

def game_ini(Handi): #initiates a new game with given handicap
    newgame = np.zeros((PLANES, BOARD, BOARD))
    newgame[2:5] = np.ones([3, BOARD, BOARD])  # free plane: initially all free. And player, legal
    if Handi == 1: # these prepare the board for the correct handicap stones
        global BOT, nBOT
        BOT = "B"
        nBOT = "W"
        newgame, resultmatrix=prediction(newgame)
        printgame(newgame, resultmatrix)
    if Handi >= 2:
        newgame[0,15,3] = 1
        newgame[2,15,3] = 0
        newgame[4,15,3] = 0
        newgame[0,3,15] = 1
        newgame[2,3,15] = 0
        newgame[4,3,15] = 0
    if Handi >= 3:
        newgame[0][15][15] = 1
        newgame[2][15][15] = 0
        newgame[4][15][15] = 0
    if Handi >= 4:
        newgame[0][3][3] = 1
        newgame[2][3][3] = 0
    if Handi == 5 or Handi == 7 or Handi == 9:
        newgame[0][6][6] = 1
        newgame[2][6][6] = 0
    if Handi >= 6:
        newgame[0][6][3] = 1
        newgame[2][6][3] = 0
        newgame[0][6][9] = 1
        newgame[2][6][9] = 0
    if Handi >= 8:
        newgame[0][3][6] = 1
        newgame[2][3][6] = 0
        newgame[0][9][6] = 1
        newgame[2][9][6] = 0
    return newgame



#### example commands ####

game = game_ini(HC)
#game = play("D4",game) #play a move
#game[column-1][line-1][color(X:0 O:1)]=0 delete a move