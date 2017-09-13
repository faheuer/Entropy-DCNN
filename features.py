# Includes all functions needed to extract features from the board and plug them in the
# input tensor. Can be modified to extract up to 30 features (current: 13)


############################### make groups & groupsize ###################################

import numpy as np


def on_board(f): # returns true if field within board limits
    for n in f:
        if n < 0 or n > 12:
            return False
    return True


def lastmoves(boardpos, X,Y): # last moves that were played
    BOARD = boardpos.shape[2]

    for a in range(BOARD):
        for b in range(BOARD):
            if boardpos[8][a][b] != 0:
                boardpos[8][a][b] -= 1
    boardpos[8][X][Y] = 4

def groupmaker(boardpos): # generates groups
    for c in (range(2)):
        thiscolour = boardpos[c].copy()  # does it once for this and once for other color
        othercolour = boardpos[1-c]

        for i in range(boardpos.shape[2]):  # creates group, first stone
            for j in range(boardpos.shape[2]):
                groupboard = np.zeros([boardpos.shape[2], boardpos.shape[2]],dtype=int)
                scount = 0
                if thiscolour[i][j] == 1:
                    thiscolour, groupboard, scount = stoneremover(thiscolour, groupboard, i, j, scount)  # adds stones & libs
                    if scount == 5:  # counts groupsize in: 1,2,3,4-5,6-10,11+
                       scount -= 1
                    if 6 <= scount <= 10:
                        scount = 5
                    if scount >= 11:
                        scount = 6
                    #boardpos[scount + 4] += groupboard  # add group in the correct size location

                    liberties, _ = libertycounter(groupboard, othercolour)
                    if liberties > 3:  # distinguishes up to 3 libs
                        liberties = 3
                    boardpos[liberties + 4] += groupboard  # add group in the correct libs location


def libertycounter(groupboard, othercolour): #counts liberties of a given group and returns them and their amount
    BOARD = othercolour.shape[1]
    libboard = np.zeros([BOARD, BOARD], dtype=int)
    liberties = 0
    for k in range(BOARD):  # assign liberties
        for l in range(BOARD):
            if groupboard[k][l] == 1:
                if k + 1 in range(BOARD) and groupboard[k + 1][l] == 0 and othercolour[k + 1][l] == 0:
                    libboard[k + 1][l] = 1
                if k - 1 >= 0 and groupboard[k - 1][l] == 0 and othercolour[k - 1][l] == 0:
                    libboard[k - 1][l] = 1
                if l + 1 in range(BOARD) and groupboard[k][l + 1] == 0 and othercolour[k][l + 1] == 0:
                    libboard[k][l + 1] = 1
                if l - 1 >= 0 and groupboard[k][l - 1] == 0 and othercolour[k][l - 1] == 0:
                    libboard[k][l - 1] = 1
    for m in range(BOARD):
        for n in range(BOARD):
            if libboard[m][n] == 1:
                liberties += 1
    return liberties, libboard

def stoneremover(thiscolour, groupboard, i, j, scount):  # takes a group from thiscolour to groupboard for further analysis

    BOARD = thiscolour.shape[1]
    groupboard[i][j] = 1
    thiscolour[i][j] = 0
    scount += 1
    if i + 1 in range(BOARD) and thiscolour[i + 1][j] == 1:
        thiscolour, groupboard, scount = stoneremover(thiscolour, groupboard, i + 1, j, scount)
    if i - 1 >= 0 and thiscolour[i - 1][j] == 1:
        thiscolour, groupboard, scount = stoneremover(thiscolour, groupboard, i - 1, j, scount)
    if j + 1 in range(BOARD) and thiscolour[i][j + 1] == 1:
        thiscolour, groupboard, scount = stoneremover(thiscolour, groupboard, i, j + 1, scount)
    if j - 1 >= 0 and thiscolour[i][j - 1] == 1:
        thiscolour, groupboard, scount = stoneremover(thiscolour, groupboard, i, j - 1, scount)
    return thiscolour, groupboard, scount

def stonekiller(onelibplane, freeplane, enemyplane, xpos, ypos): #takes stones off board when killed
    BOARD = enemyplane.shape[1]

    koinfo = 0
    if xpos + 1 in range(BOARD) and enemyplane[xpos + 1, ypos] == 1 and onelibplane[ xpos + 1, ypos] == 1:  # remove captured group of enemy
        _, killed, koinfo = stoneremover(enemyplane, np.zeros([BOARD, BOARD],dtype=int), xpos + 1, ypos, 0)
        freeplane += killed  # restore free field plane
    if xpos - 1 >= 0 and enemyplane[xpos - 1, ypos] == 1 and onelibplane[xpos - 1, ypos] == 1:  # if neigboring group is in 1 liberty plane
        _, killed, koinfo = stoneremover(enemyplane, np.zeros([BOARD, BOARD],dtype=int), xpos - 1, ypos, 0)  # remove group from enemy
        freeplane += killed
    if ypos + 1 in range(BOARD) and enemyplane[xpos, ypos + 1] == 1 and onelibplane[xpos, ypos + 1] == 1:
        _, killed, koinfo = stoneremover(enemyplane, np.zeros([BOARD, BOARD],dtype=int), xpos, ypos + 1, 0)
        freeplane += killed
    if ypos - 1 >= 0 and enemyplane[xpos, ypos - 1] == 1 and onelibplane[xpos, ypos - 1] == 1:
        _, killed, koinfo = stoneremover(enemyplane, np.zeros([BOARD, BOARD],dtype=int), xpos, ypos - 1, 0)
        freeplane += killed
    if koinfo == 1:     # ko legalness for sensibleness
        return True, killed
    else: return False, None

def libsafter(boardpos):
    BOARD = boardpos.shape[2]

    for a in range(BOARD):
        for b in range(BOARD):
            thiscolour = boardpos[1].copy()
            othercolour = boardpos[0].copy()
            groupboard = np.zeros([BOARD, BOARD], dtype=int)
            if boardpos[2][a][b] == 1: # if free
                _, groupboard, _ = stoneremover(thiscolour, groupboard, a, b, 0) #create group for next move
                _, _ = stonekiller(boardpos[5], np.zeros([BOARD,BOARD],dtype=int), othercolour, a, b)  # kill stones in temporary othercolour plane

                liberties, _ = libertycounter(groupboard, othercolour)                 #counts liberties of this group
                if liberties > 6:  # distinguishes up to 6 libs
                    liberties = 6
                if liberties == 0:  # if move has 0 liberties, add as illegal (sensibleness)
                    boardpos[4][a][b] = 0
                if liberties < 3 and liberties != 0:  # distinguishes 1 and 2 libs
                    boardpos[liberties + 10][a][b] = 1 # add move in correct next move libs plane

def bigeyechecker(thiscolour, x, y, ecount): # checks if missing field in eye is actually an eye
    BOARD = thiscolour.shape[1]

    if ecount > 7:
        return thiscolour, ecount

    thiscolour[x][y] = 1
    ecount += 1

    if x + 1 in range(BOARD) and thiscolour[x + 1][y] == 0:
        thiscolour, ecount = bigeyechecker(thiscolour, x + 1, y, ecount)
    if x - 1 >= 0 and thiscolour[x - 1][y] == 0:
        thiscolour, ecount = bigeyechecker(thiscolour, x - 1, y, ecount)
    if y + 1 in range(BOARD) and thiscolour[x][y + 1] == 0:
        thiscolour, ecount = bigeyechecker(thiscolour, x, y + 1, ecount)
    if y - 1 >= 0 and thiscolour[x][y - 1] == 0:
        thiscolour, ecount = bigeyechecker(thiscolour, x, y - 1, ecount)
    return thiscolour, ecount


def eyechecker(boardpos, v, w):  # checks if empty field is eye and makes eyes insensible moves
    BOARD = boardpos.shape[2]

    if boardpos[4][v][w] == 1:
        freecounter = 0  # one kosumi field is allowed to be free to still be an eye
        sidecounter = 0  # eyes on the side have special properties
        fieldx = []
        fieldy = []
        if v + 1 not in range(BOARD) or boardpos[1][v + 1][w] == 1:
            if v - 1 < 0 or boardpos[1][v - 1][w] == 1:
                if w + 1 not in range(BOARD) or boardpos[1][v][w + 1] == 1:
                    if w - 1 < 0 or boardpos[1][v][w - 1] == 1:
                        if v + 1 in range(BOARD) and w + 1 in range(BOARD):
                            if boardpos[1][v + 1][w + 1] == 0:
                                freecounter += 1
                                fieldx.append(v + 1)
                                fieldy.append(w + 1)
                        else: sidecounter += 1
                        if v + 1 in range(BOARD) and w - 1 >= 0:
                            if boardpos[1][v + 1][w - 1] == 0:
                                freecounter += 1
                                fieldx.append(v + 1)
                                fieldy.append(w - 1)
                        else: sidecounter += 1
                        if v - 1 >= 0 and w - 1 >= 0:
                            if boardpos[1][v - 1][w - 1] == 0:
                                freecounter += 1
                                fieldx.append(v - 1)
                                fieldy.append(w - 1)
                        else: sidecounter += 1
                        if v - 1 >= 0 and w + 1 in range(BOARD):
                            if boardpos[1][v - 1][w + 1] == 0:
                                freecounter += 1
                                fieldx.append(v - 1)
                                fieldy.append(w + 1)
                        else: sidecounter += 1
                        if (sidecounter == 0 and freecounter <= 1) or (freecounter == 0):
                            boardpos[4][v][w] = 0  # if full eye, then make it insensible

                        else: #potential false eye
                            edgecount = 0   # counts non-eye fields
                            _, ecount = bigeyechecker(boardpos[1].copy(),fieldx[0],fieldy[0], 0)
                            if ecount > 7: # neighbouring eye is not too big/open space
                                edgecount += 1  # it is actually an eye, make it insensible
                            if len(fieldx) > 1: # check next empty field if existant
                                _, ecount = bigeyechecker(boardpos[1].copy(), fieldx[1], fieldy[1], 0)
                                if ecount > 7:
                                    edgecount += 1
                                if len(fieldx) > 2:
                                    _, ecount = bigeyechecker(boardpos[1].copy(), fieldx[2], fieldy[2], 0)
                                    if ecount > 7:
                                        edgecount += 1
                                    if len(fieldx) > 3:
                                        _, ecount = bigeyechecker(boardpos[1].copy(), fieldx[3], fieldy[3], 0)
                                        if ecount > 7:
                                            edgecount += 1
                            if (edgecount <= 1 and sidecounter == 0) or edgecount == 0:
                                boardpos[4][v][w] = 0  # if one or less non-eyes, this eye is real





def sensibleness(boardpos, ko, koinfo):  # illegal are: ko, don't fill own eyes
    BOARD = boardpos.shape[2]

    if ko:
        boardpos[4] -= koinfo
        if -1 in boardpos[4]:  # if already marked illegal, undo
            boardpos[4] += koinfo


    for v in range(BOARD):
        for w in range(BOARD):
            eyechecker(boardpos, v, w) # don't fill own eyes


################### functions to read out ladders #############################


def ladderdeadstones(enemyplane, ownplane, xpos, ypos): #kills stones, only for ladders. ownplane = killing played move
    BOARD = ownplane.shape[1]


    if xpos + 1 in range(BOARD) and enemyplane[xpos + 1, ypos] == 1:
        _, group, _ = stoneremover(enemyplane.copy(), np.zeros([BOARD,BOARD],dtype=int), xpos + 1, ypos, 0)
        liberties, _ = libertycounter(group, ownplane)
        if liberties == 0:
            stoneremover(enemyplane, np.zeros([BOARD, BOARD],dtype=int), xpos + 1, ypos, 0)
    if xpos - 1  >= 0 and enemyplane[xpos - 1, ypos] == 1:
        _, group, _ = stoneremover(enemyplane.copy(), np.zeros([BOARD,BOARD],dtype=int), xpos - 1, ypos, 0)
        liberties, _ = libertycounter(group, ownplane)
        if liberties == 0:
            stoneremover(enemyplane, np.zeros([BOARD, BOARD],dtype=int), xpos - 1, ypos, 0)
    if ypos + 1 in range(BOARD) and enemyplane[xpos, ypos + 1] == 1:
        _, group, _ = stoneremover(enemyplane.copy(), np.zeros([BOARD,BOARD],dtype=int), xpos, ypos + 1, 0)
        liberties, _ = libertycounter(group, ownplane)
        if liberties == 0:
            stoneremover(enemyplane, np.zeros([BOARD, BOARD],dtype=int), xpos, ypos + 1, 0)
    if ypos - 1  >= 0 and enemyplane[xpos, ypos - 1] == 1:
        _, group, _ = stoneremover(enemyplane.copy(), np.zeros([BOARD,BOARD],dtype=int), xpos, ypos - 1, 0)
        liberties, _ = libertycounter(group, ownplane)
        if liberties == 0:
            stoneremover(enemyplane, np.zeros([BOARD, BOARD],dtype=int), xpos, ypos - 1, 0)

    return enemyplane

def captureladder(laddergroup, thiscolour, othercolour, laddersteps, iniladder, first):  # plays to reduce ladder libs to 1. thiscolour = col of the ladder
    BOARD = thiscolour.shape[1]

    laddersteps += 1
    if laddersteps == 1:  # remeber initial ladder group
        iniladder += laddergroup

    _ , libboard = libertycounter(laddergroup, othercolour)
    for v in range(BOARD):
        for w in range(BOARD):
            if libboard[v][w] == 1:
                death, laddersteps, iniladder, first = captureladderintent(laddergroup, thiscolour.copy(), othercolour.copy(), laddersteps, iniladder, first, v, w)  # new function it is used very frequently
                if death:
                    return True, laddersteps, iniladder, first
    return False, laddersteps, iniladder, None    # ladder escapes if no captures were found

def captureladderintent(laddergroup, thiscolour, othercolour, laddersteps, iniladder, first, v, w):
    BOARD = thiscolour.shape[1]

    othercolour[v][w] = 1
    #thiscolour = ladderdeadstones(thiscolour, othercolour, v, w)
    _, capturestones, _ = stoneremover(othercolour.copy(), np.zeros([BOARD, BOARD], dtype=int), v, w, 0)  # check if it is legal to play this move (0libs)
    liberties, _ = libertycounter(capturestones, thiscolour)
    if liberties >= 1:  # if legal
        death, laddersteps, iniladder, first = escapeladder(laddergroup, thiscolour, othercolour, laddersteps, iniladder, first)
        if death:
            first = [v, w]
            return True, laddersteps, iniladder, first  # ladder dies
    return False, laddersteps, iniladder, first



def escapeladder(laddergroup, thiscolour, othercolour, laddersteps, iniladder, first):  # plays to extend the laddergroup. thiscolour = col of the ladder
    BOARD = thiscolour.shape[1]

    laddersteps += 1
    if laddersteps == 1:
        iniladder += laddergroup   # remeber initial ladder group

    liberties, libboard = libertycounter(laddergroup, othercolour)
    if liberties > 1:
        return False, laddersteps, iniladder, first  # ladder escapes
    else:   # normal case: only one liberty for escaping ladder
        move = np.unravel_index(libboard.argmax(), libboard.shape) # returns tuple of the last liberty move
    _, laddergroup, _ = stoneremover(thiscolour.copy(), np.zeros([BOARD,BOARD],dtype=int),move[0],move[1],0)
    thiscolour[move[0]][move[1]] = 1
    # HERE REMOVE KILLED
    liberties, _ = libertycounter(laddergroup, othercolour)
    if liberties == 2:  # play next move
        death, laddersteps, iniladder, _ = captureladder(laddergroup, thiscolour, othercolour, laddersteps, iniladder, first)
        if death:
            death, laddersteps, iniladder, first = capturesurrounders(laddergroup, thiscolour, othercolour, laddersteps, iniladder, first)
            if death:  # if taking stones from the surrounders doesnt prevent ladder dying
                return True, laddersteps, iniladder, None
            else: return False, laddersteps, iniladder, first  # if taking off surrounders makes group live
        else:
            first = [move[0], [move[1]]]  # escaping move
            return False, laddersteps, iniladder, first

    elif liberties <= 1:
        thiscolour[move[0]][move[1]] = 0  # don't make the move
        laddergroup[move[0]][move[1]] = 0
        death, laddersteps, iniladder, first = capturesurrounders(laddergroup, thiscolour, othercolour, laddersteps, iniladder, first) # try capturing surrounding stones instead
        if death:  # if taking stones from the surrounders doesnt prevent ladder dying
            return True, laddersteps, iniladder, None
        else:
            return False, laddersteps, iniladder, first  # if taking off surrounders makes group live
    elif liberties >= 3:
        laddergroup[move[0]][move[1]] = 0
        thiscolour[move[0]][move[1]] = 0
        first = [move[0], [move[1]]]  # escaping move
        return False, laddersteps, iniladder, first    # ladder escapes

def capturesurrounders(laddergroup, thiscolour, othercolour, laddersteps, iniladder, first):  # instad of extending group, try capturing surrounding stones. thiscolour = col of the ladder
    BOARD = thiscolour.shape[1]

    laddersteps += 1
    othercolour_capturers = othercolour.copy()  # to identify laddersurrounding stones and not try them twice
    _, surrounders = libertycounter(laddergroup, np.zeros([BOARD,BOARD],dtype=int))  # surrounding stones of the ladder
    for a in range(BOARD):
        for b in range(BOARD):
            if surrounders[a][b] == 1:
                if othercolour_capturers[a][b] == 1:
                    _, capturestones, capturecount = stoneremover(othercolour_capturers, np.zeros([BOARD, BOARD],dtype=int), a, b, 0)  # find groups of othercol
                    captureliberties, capturelibboard = libertycounter(capturestones, thiscolour)
                    if captureliberties == 1:       # that have only 1 liberty (0 is already ruled out in captureladder)
                        if capturecount >=2:  # if group with 2 or more stones captured: ladder is broken
                            if laddersteps == 2: first = np.unravel_index(capturelibboard.argmax(), capturelibboard.shape)  # 2 because 1 has already been "made" in escapeladder
                            return False, laddersteps, iniladder, first
                        else:
                            thiscolour += capturelibboard  # play to kill off 1 surrounding stone
                            othercolour -= capturestones  # take the stone off the board
                            if laddersteps == 2: first =  np.unravel_index(capturelibboard.argmax(), capturelibboard.shape)
                            death, laddersteps, iniladder, _ = captureladder(laddergroup, thiscolour, othercolour, laddersteps, iniladder, first)
                            if not death:
                                return False, laddersteps, iniladder, first
    return True, laddersteps, iniladder, None  # if no stone killing prevents the ladder from dying


def laddermaker(boardpos):
    BOARD = boardpos.shape[2]

    onelib = boardpos[5].copy()  # this is used and edited for 1-liberty groups
    twolib = boardpos[6].copy()  # and 2-liberty groups
    colourA = boardpos[0]         # to play out the ladder, make copy (below)
    colourB = boardpos[1]

    # case 1: make ladder by taking 1 liberty

    for v in range(BOARD):
        for w in range(BOARD):
            if twolib[v][w] == 1:
                if colourA[v][w]==1:
                    _ , laddergroup, _ = stoneremover(colourA.copy(), np.zeros([BOARD,BOARD],dtype=int),v,w,0)
                else: _ , laddergroup, _ = stoneremover(colourB.copy(), np.zeros([BOARD,BOARD], dtype=int),v,w,0)
                twolib -= laddergroup
                if colourA[v][w]==1:
                    death, laddersteps, iniladder, first = captureladder(laddergroup,colourA.copy(),colourB.copy(), 0, np.zeros([BOARD,BOARD],dtype=int),None)
                else: death, laddersteps, iniladder, first = captureladder(laddergroup,colourB.copy(),colourA.copy(), 0, np.zeros([BOARD,BOARD],dtype=int),None)
                if laddersteps >= 5:  # only regard ladders of reasonable size
                    if first: iniladder[first[0],first[1]]=1
                    if death:
                        boardpos[9] += iniladder
                    else: boardpos[10] += iniladder


    # case 2: running out by extending group

    for x in range(BOARD):
        for y in range(BOARD):
            if onelib[x][y] == 1:
                if colourA[x][y] == 1:
                    _ , laddergroup, _ = stoneremover(colourA.copy(), np.zeros([BOARD,BOARD], dtype=int),x,y,0)
                else: _ , laddergroup, _ = stoneremover(colourB.copy(), np.zeros([BOARD,BOARD], dtype=int),x,y,0)
                onelib -= laddergroup
                if colourA[x][y]==1:
                    death, laddersteps, iniladder, first = escapeladder(laddergroup,colourA.copy(),colourB.copy(), 0, np.zeros([BOARD,BOARD], dtype=int), None)
                else: death, laddersteps, iniladder, first = escapeladder(laddergroup, colourB.copy(), colourA.copy(), 0, np.zeros([BOARD, BOARD],dtype=int), None)
                if laddersteps >= 7:  # only regard ladders of reasonable size (all tried moves are steps)
                    if first: iniladder[first[0],first[1]]=1
                    if death:
                        boardpos[9] += iniladder
                    else: boardpos[10] += iniladder



def extract(boardpos, xpos, ypos, col):   # this plays the stone and removes kills
    BOARD = boardpos.shape[2]

    boardpos[0, xpos, ypos] = 1  # add 1 in ourplane

    if col =="B":
        boardpos[3] = np.ones([BOARD, BOARD],dtype=int)  # we are b (means: w to play position)
    else:
        boardpos[3] = np.zeros([BOARD, BOARD],dtype=int)  # we are w


    ko, koinfo = stonekiller(boardpos[5], boardpos[2], boardpos[1], xpos, ypos)  # kill stones if necessary
    boardpos[2, xpos, ypos] = 0  # adjust free field
    boardpos[4] = boardpos[2]  #only free are sensible
    boardpos[5:8] = np.zeros([3, boardpos.shape[2], boardpos.shape[2]])  # clears groupsize planes
    boardpos[9:13] = np.zeros([4, boardpos.shape[2], boardpos.shape[2]])
    groupmaker(boardpos)  # fills layers for group sizes, libs for b and w
    libsafter(boardpos)   # fills liberties after move planes
    lastmoves(boardpos, xpos, ypos)
    sensibleness(boardpos, ko, koinfo)
    laddermaker(boardpos)

    return boardpos


