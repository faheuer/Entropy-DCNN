# This file takes files in the sgf go format and transforms them to input tensors with the help of features.py.
# It is heavily dependant on multiprocessing.

import numpy as np
import glob
import features
from multiprocessing import Pool, Process, Lock
import multiprocessing
import os.path

BOARD = 19
SAVEFILE = 20
SAVEBIGFILE = 40000

kgsdatain = np.array([],dtype=int)            # NN input data (3D elements)
kgsdataout = np.array([],dtype=int)           # NN feedback (2D elements)
kgsdatawin = np.array([],dtype=int)           # NN feedback (2D elements)
gamecounter = 0                       # counts nr. of games for runtime optimization
newin = np.array([])
newout = np.array([])
newwin = np.array([])
np.save('/media/falk/6,0 TB Volume/19 badukmovies/kgsdatain_alllayer.npy', newin) #can be replaced with goal destination
np.save('/media/falk/6,0 TB Volume/19 badukmovies/kgsdataout_alllayer.npy', newout)
np.save('/media/falk/6,0 TB Volume/19 badukmovies/kgsdatawin_alllayer.npy', newwin)
alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s"]
initboardpos = np.zeros([13, BOARD, BOARD],dtype=int)        # 1D:game positions, 2D: b&w&free&B=us?&sesible&libs(5-7)
                                    # &lastmove(8)&laddery/n(9-10)&libsafter(11-12) resp. boards, 3.,4.:board itself

initboardpos[2:5] = np.ones([3, BOARD, BOARD])     # free plane: initially all free. And we play black

# The third plane is needed to recognize the size of the board because of e.g. zero-padding

############################### save kgsdata ###################################





def process(file): # used by multiprocessor. Tensors are stored multiple times to reduce RAM usage.
    global gamecounter, kgsdatain, kgsdataout, kgsdatawin

    print file
    gamefile = open(file, 'r')
    gamecounter += 1
    if gamecounter == SAVEFILE:
        lock.acquire()
        newin = np.load('/media/falk/6,0 TB Volume/19 badukmovies/kgsdatain_alllayer.npy')
        newout = np.load('/media/falk/6,0 TB Volume/19 badukmovies/kgsdataout_alllayer.npy')
        newwin = np.load('/media/falk/6,0 TB Volume/19 badukmovies/kgsdatawin_alllayer.npy')
        newin = np.concatenate([newin, kgsdatain]) if newin.size else kgsdatain
        newout = np.concatenate([newout, kgsdataout]) if newout.size else kgsdataout
        newwin = np.concatenate([newwin, kgsdatawin]) if newwin.size else kgsdatawin
        kgsdatain = np.array([])
        kgsdataout = np.array([])
        kgsdatawin = np.array([])
        print("Spielblock gespeichert")
        if len(newin) >= SAVEBIGFILE:
            metacounter = 0
            while os.path.isfile('/media/falk/6,0 TB Volume/19 badukmovies/pro/kgsdatain_alllayer_'+ str(metacounter)+'.npy'):
                metacounter += 1
            newinstr = '/media/falk/6,0 TB Volume/19 badukmovies/pro/kgsdatain_alllayer_'+ str(metacounter)+'.npy'
            newoutstr = '/media/falk/6,0 TB Volume/19 badukmovies/pro/kgsdataout_alllayer_'+ str(metacounter)+'.npy'
            newwinstr = '/media/falk/6,0 TB Volume/19 badukmovies/pro/kgsdatawin_alllayer_'+ str(metacounter)+'.npy'
            # this is the final storage location
            np.save(newinstr, newin)
            np.save(newoutstr, newout)
            np.save(newwinstr, newwin)
            newout = np.array([])
            newin = np.array([])
            newwin = np.array([])
            print("--------------- META Spielblock gespeichert--------------------")
        np.save('/media/falk/6,0 TB Volume/19 badukmovies/kgsdatain_alllayer.npy', newin)
        np.save('/media/falk/6,0 TB Volume/19 badukmovies/kgsdataout_alllayer.npy', newout)
        np.save('/media/falk/6,0 TB Volume/19 badukmovies/kgsdatawin_alllayer.npy', newwin)
        lock.release()
        gamecounter = 0


    gamefile = gamefile.read()
    gamefile = gamefile.replace("\n","")
    gamefile = gamefile.replace("(;","")
    game = gamefile
    boardpos = initboardpos.copy()

    if "HA[" in gamefile:            # if Handycap
        handycap = int(gamefile.split("HA[",1)[1][0]) #write down Handycap
        if handycap>=2:
            (boardpos[0,15,3], boardpos[0,3,15])=(1,1)
            (boardpos[2, 15, 3], boardpos[2, 3, 15]) = (0, 0)
        if handycap>=3:
            boardpos[0, 15, 15] = 1
            boardpos[2, 15, 15] = 0
        if handycap>=4:
            boardpos[0,3,3]=1
            boardpos[2, 3, 3] = 0
        if handycap==5:
            boardpos[0, 9, 9] = 1
            boardpos[2, 9, 9] = 0
        if handycap>=6:
            (boardpos[0, 9, 3], boardpos[0, 9, 15]) = (1, 1)
            (boardpos[2, 9, 3], boardpos[0, 9, 15]) = (0, 0)
        if handycap==7:
            boardpos[0, 9, 9] = 1
            boardpos[2, 9, 9] = 0
        if handycap>=8:
            (boardpos[2, 3, 9], boardpos[0, 15, 9]) = (0, 0)
        if handycap==9:
            boardpos[0, 9, 9] = 1
            boardpos[2, 9, 9] = 0


# main sgf disassembly begins here

    for sig, col, y, x in zip(game, game[1:], game[3:], game[4:]):      # structure: ;B[ic]

                ######################## store prev board, feedback ##########################
        if sig == ";":
            if x not in alphabet: break                                 # error or pass: leave game
            if y not in alphabet: break                                 # error or pass: leave game
            if col not in ("B", "W"): break                             # error: leave game
            xpos = alphabet.index(x)
            ypos = alphabet.index(y)
            storeboardpos=boardpos[None,:]
            kgsdatain = np.concatenate([kgsdatain, storeboardpos]) if kgsdatain.size else storeboardpos
                                                    # store to database (uses boardpos from last iteration)
            boardpos2 = boardpos[0:2].copy()        # swap board for agent to always have color of plane 1
            boardpos2[1] = boardpos[0]              # 1st agent is b, 2nd w, 3rd b etc.
            boardpos[0] = boardpos[1]
            boardpos[1] = boardpos2[1]

            onemove = np.zeros([1, BOARD, BOARD],dtype=int) # add one in feedback board
            onemove[0, xpos, ypos] = 1
            onewin = np.array([1])

            if "RE[B" in game:  # b wins = 1
                onewin[0] = 1
            else:
                onewin[0] = 0

            kgsdatawin = np.concatenate([kgsdatawin, onewin]) if kgsdatawin.size else onewin # we lose
            kgsdataout = np.concatenate([kgsdataout, onemove]) if kgsdataout.size else onemove
                ############################# add new move #################################

            boardpos = features.extract(boardpos, xpos, ypos, col) # modifies all layers in boardpos

def init(l):
    global lock
    lock = l

lock = Lock()
pool = Pool(multiprocessing.cpu_count(), initializer=init, initargs=(lock,))                        # Create a multiprocessing Pool
pool.map(process, glob.glob('Data/badukmovies/*/*/*.sgf'))                                                     # proces data_inputs iterable with pool
pool.close()
pool.join()



#show the planes with stone colour and the output
print kgsdatain[218][0]
print kgsdatain[218][1]
print kgsdataout[218]


