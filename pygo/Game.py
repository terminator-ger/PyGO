from os import remove
import random
import numpy as np
from enum import Enum
from datetime import datetime
from playsound import playsound
from sgfmill import sgf
from utils.color import N2C, C2N
from typing import List, Tuple
import pdb

class GameState(Enum):
    RUNNING = 0
    NOT_STARTED = 1


class Game:
    def __init__(self):
        self.state = np.ones((19,19),dtype=np.int64)*2
        self.last_color = 2
        self.last_x = -1
        self.last_y = -1
        # 0 = white
        # 1 = black
        # 2 = empty
        self.GS = GameState.NOT_STARTED
        self.sgf = None
        self.sgf_node = None
       
    def startNewGame(self, size=19):
        self.sgf = sgf.Sgf_game(size=size)
        self.sgf_node = self.sgf.get_root()
        self.GS = GameState.RUNNING
       
    def endGame(self):
        if self.GS == GameState.RUNNING:
            cur_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
            with open('{}.sgf'.format(cur_time), 'wb') as f:
                f.write(self.sgf.serialise())

            self.GS = GameState.NOT_STARTED
            self.sgf = None

    def set(self, stone, coords):
        x = coords[0]
        y = coords[1]
        c_str = stone

        self.sgf_node.set_move(c_str.lower(),(x,y))
        self.sgf_node = self.sgf.extend_main_sequence()
        self.last_x = x
        self.last_y = y
        print('{}: {}-{}'.format(c_str, x+1, y+1))
        self.state[x,y] = C2N(c_str)
        self.last_color = C2N(c_str)

    def nextMove(self):
        if self.last_color == 0:
            return 1
        elif self.last_color == 1:
            return 0
        else:
            return None


    def updateStateNoChecks(self, state):
        state = state.reshape(19,19)
        idx = np.argwhere(np.abs(self.state-state)>0)
        if len(idx) > 0:
            x = idx[0,0]
            y = idx[0,1]
            c = state[x,y]
            rnd = random.randint(1,5) 
            playsound('sounds/stone{}.wav'.format(rnd))
            c_str = N2C(c)

            print('{}: {}-{}'.format(c_str, x+1, y+1))
            self.last_x = x
            self.last_y = y
            self.state = state
            self.last_color = c
            return [c_str.lower(), x, y]



    def updateState(self, state):
        '''
        input detected state from classifier
        '''
        if self.GS == GameState.RUNNING:   
            return self.updateStateWithChecks(state)
        else:
            return self.updateStateNoChecks(state)


    def whichMovesAreInTheGameTree(self, state) -> Tuple[List,List]:
        state = state.reshape(19,19)
        idx = np.argwhere(np.abs(self.state-state)>0)
        if self.GS == GameState.RUNNING:
            X = idx[:,0]
            Y = idx[:,1]
            isInTree = []
            notInTree = []
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                c = state[x,y]
                c_str = N2C(c)
                seq = self.sgf.get_main_sequence()[:-2]
                for node in seq:
                    if node.properties()[-1] == c_str and (x,y) == node.get(c_str):
                        isInTree.append((c_str, (x,y)))
                        break
                # not found
                notInTree.append((c_str,(x,y)))
            return (isInTree, notInTree)
        else:
            return ([],[])


    
    def updateStateWithChecks(self, state):
        state = state.reshape(19,19)
        diff = np.count_nonzero(np.abs(self.state-state))
        idx = np.argwhere(np.abs(self.state-state)>0)
        seq = self.sgf.get_main_sequence()
        isInTree, notInTree = self.whichMovesAreInTheGameTree(state)

        if diff == 1:
            # either one stone added -> regular move or
            # last stone removed
            c_str, (x,y) = notInTree.pop(0)
            c = C2N(c_str)

            if c != self.last_color and c == 2:
                #check wether the move is the last in the tree:
                if len(seq) > 0:
                    last = seq[-2]
                    last_move_color = last.properties()
                    last_move_pos = last.get(last_move_color[-1])
                    if last_move_pos  == (x,y):
                        print('Undo Last Move')
                        if len(seq) >=3:
                            self.sgf_node.reparent(last.parent, -1)
                            n = seq[-3].properties()[-1]
                            self.last_color = C2N(n)
                        else:
                            self.sgf_node.reparent(self.sgf_node.parent, -1)
                            self.last_color = 2

                        # clear state
                        self.state[x,y] = 2
                        return ["", -1, -1]
                else:
                    # cannot undo moves when there are no moves recorded
                    return ["", -1, -1]

            elif c != self.last_color and c in [0,1]:
                # new move
                rnd = random.randint(1,5) 
                playsound('sounds/stone{}.wav'.format(rnd))
                self._setStone(x, y, c_str)
                return [c_str.lower(), x, y]

        elif diff == 2:
            #either one captured stone or we moved the last one
            if len(isInTree) == 1:
                # we moved the last stone
                self.sgf_node.reparent(self.sgf_node.parent, -1)
                rnd = random.randint(1,5) 
                playsound('sounds/stone{}.wav'.format(rnd))
                self._captureStone(isInTree)
                c_str, (x,y) = notInTree.pop(0)
                self._setStone(x, y, c_str)
                return [c_str.lower(), x, y]

            else:
                # we need at least one removed stone and one added stone..
                if np.sum([x[0] != 'E' for x in notInTree]) == 1:
                    nm = self.nextMove()
                    idx = np.argwhere([C2N(x[0]) == nm for x in notInTree])
                    if len(idx) > 0:
                        c_str, (x,y) = notInTree.pop(idx.squeeze())
                        # stones that was captured
                        # upon faulty state we would simply copy it ...
                        # TODO: implement capute check and ko check...
                        self._captureStone(x, y)
                        c_str, (x,y) = notInTree.pop(0)
                        self._setStone(x, y, c_str)
                        playsound('sounds/capturing.wav')
                        return [c_str.lower(), x, y]
        else:
            if np.sum([x[0] != 'E' for x in notInTree]) == 1:
                nm = self.nextMove()
                idx = np.argwhere([C2N(x[0]) == nm for x in notInTree])
                if len(idx) > 0:
                    move = notInTree.pop(idx.squeeze())
                    for [_,removed] in notInTree:
                        self._captureStone(removed[0],removed[1])

                    # stones where captured
                    c_str = move[0]
                    (x,y) = move[1]
                    self._setStone(x, y, c_str)
                    playsound('sounds/capturing.wav')
                    return [c_str.lower(), x, y]    
    
    def _setStone(self, x:int, y:int, c_str:str) -> None:
        c = C2N(c_str)
        self.sgf_node.set_move(c_str.lower(),(x,y))
        self.sgf_node = self.sgf.extend_main_sequence()
        self.last_x = x
        self.last_y = y
        print('{}: {}-{}'.format(c_str, x+1, y+1))
        # upon faulty state we would simply copy it ...
        # TODO: implement capute check and ko check...

        self.state[x,y] = c
        self.last_color = c

    def _captureStone(self, x: int, y: int) -> None:
        print('Capture: {}-{}'.format(x+1, y+1))
        self.state[x,y] = 2


