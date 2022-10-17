import enum
from os import remove
import random
from re import A
import numpy as np

from enum import Enum
from datetime import datetime
from playsound import playsound

from sgfmill import sgf
from typing import List, TextIO, Tuple, Optional
import pdb
from pudb import set_trace
import cv2
import logging

from pygo.utils.image import toByteImage
from pygo.utils.color import N2C, C2N, COLOR
from pygo.utils.debug import DebugInfo, DebugInfoProvider
from pygo.utils.typing import Move, NetMove, GoBoardClassification


class GameState(Enum):
    RUNNING = 0
    NOT_STARTED = 1


class Game(DebugInfoProvider):
    def __init__(self):
        super().__init__()
        self.last_color = 2
        self.last_x = -1
        self.last_y = -1
        self.board_size = 19
        self.state: GoBoardClassification = np.ones((self.board_size, 
                                                     self.board_size),dtype=int)*2
        # 0 = white
        # 1 = black
        # 2 = empty
        self.GS : GameState = GameState.NOT_STARTED
        self.sgf = None
        self.sgf_node = None
        self.manualMoves = []

    def getCurrentState(self) -> GoBoardClassification:
        return self.state
       
    def startNewGame(self, size=19) -> None:
        self.sgf = sgf.Sgf_game(size=size)
        self.sgf_node = self.sgf.get_root()
        self.GS = GameState.RUNNING
        self.state = np.ones((self.board_size, self.board_size), dtype=int)*2
        self.board_size = size
        self.last_x = -1
        self.last_y = -1
        self.last_color = 2
        self.manualMoves = []
       
    def saveGame(self, file: TextIO = None) -> None:
        if self.GS == GameState.RUNNING:
            if file is None:
                cur_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
                file = '{}.sgf'.format(cur_time)
                with open(file, 'wb') as f:
                    f.write(self.sgf.serialise())
            else:
                file.write(self.sgf.serialise())

            self.GS = GameState.NOT_STARTED
            self.sgf = None

    def _test_set(self, stone, coords) -> None:
        x = coords[0]
        y = coords[1]
        c_str = stone

        self.sgf_node.set_move(c_str.lower(),(x,y))
        self.sgf_node = self.sgf.extend_main_sequence()
        self.last_x = x
        self.last_y = y
        logging.info('{}: {}-{}'.format(c_str, x+1, y+1))
        self.state[x,y] = C2N(c_str)
        self.last_color = C2N(c_str)

    def nextMove(self) -> Optional[int]:
        if self.last_color == 0:
            return 1
        elif self.last_color == 1:
            return 0
        else:
            #upon init black starts
            # TODO: unless we have handicap
            return 1


    def updateStateNoChecks(self, state) -> NetMove:
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

    def applyManualMoves(self, state: GoBoardClassification ) -> GoBoardClassification:
        for (c, (x,y)) in self.manualMoves:
            state[x,y] = c
        return state

    def _unravel(self, x):
        return x.reshape(self.board_size, self.board_size)

    def updateState(self, state) -> Tuple[List[Move], List[Move]]:
        '''
        input detected state from classifier
        '''
        state = self.applyManualMoves(self._unravel(state))
        if self.GS == GameState.RUNNING:   
            return self.updateStateWithChecks(state)
        else:
            return self.updateStateNoChecks(state)


    def whichMovesAreInTheGameTree(self, state) -> Tuple[List[Move],List[Move]]:
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


    
    def updateStateWithChecks(self, state) -> NetMove:
        state = self._check_move_validity(state)
        logging.debug(state)

        diff = np.count_nonzero(np.abs(self.state-state))
        idx = np.argwhere(np.abs(self.state-state)>0)
        seq = self.sgf.get_main_sequence()

        isInTree, notInTree = self.whichMovesAreInTheGameTree(state)
        
        logging.debug('{} different moves found'.format(diff))
        if diff == 1:
            # either one stone added -> regular move or
            # last stone removed
            
            c_str, (x,y) = notInTree.pop(0)
            c = C2N(c_str)

            if c != self.last_color and c == 2:
                # Disable undo for live games -> increase stability 
                self._undoLastMove(x, y, c_str)
                return ["", -1, -1]

            elif c != self.last_color and c in [0,1]:
                # new move
                rnd = random.randint(1,5) 
                playsound('sounds/stone{}.wav'.format(rnd))
                self._setStone(x, y, c_str)
                return [c_str.lower(), x, y]

        #elif diff == 2 and len(isInTree) == 1:
        #    # we moved the last stone
        #    self.sgf_node.reparent(self.sgf_node.parent, -1)
        #    rnd = random.randint(1,5) 
        #    playsound('sounds/stone{}.wav'.format(rnd))

        #    idx = np.argwhere([C2N(x[0]) == 'E' for x in notInTree])
        #    c_str, (x,y) = isInTree.pop(0)
        #    self._undoLastMove(x, y, c_str)
        #    c_str, (x,y) = notInTree.pop(0)
        #    self._setStone(x, y, c_str)
        #    return [c_str.lower(), x, y]

        else:
            if np.sum([x[0] != 'E' for x in notInTree]) == 1:
                nm = self.nextMove()
                idx = np.argwhere([C2N(x[0]) == nm for x in notInTree])
                if len(idx) > 0:
                    move = notInTree.pop(idx.squeeze())

                    for [color, removed] in notInTree:
                            self._captureStone(removed[0],removed[1])

                    # stones where captured
                    c_str = move[0]
                    (x,y) = move[1]
                    self._setStone(x, y, c_str)
                    playsound('sounds/capturing.wav')
                    return [c_str.lower(), x, y]    
                else:
                    #last stone was moved
                    idx = np.argwhere([C2N(x[0]) == self.last_color for x in notInTree])
                    move = notInTree.pop(idx.squeeze())
                    [c_str, removed] = notInTree.pop(0)
                    self._undoLastMove(removed[0],removed[1], c_str)
                    c_str = move[0]
                    (x,y) = move[1]
                    self._setStone(x, y, c_str)
                    playsound('sounds/stone1.wav')
                    return [c_str.lower(), x, y]

    def setManual(self, x:int ,y:int) -> None:
        ''' 
            Manual override for single moves
            Fallback in case some detection does not work als intended
        '''
        if self.state[x,y] != 2:
            # delete existing move
            logging.debug("Remove Stone at {} {}".format(x,y))
            self.manualMoves.append([2, (x,y)])
        else:
            # add stone
            c = self.nextMove()
            self._setStone(x, y, N2C(c))
            #save move in overwrite state
            self.manualMoves.append([c, (x,y)])
            logging.debug("Manual overwrite: Adding {} Stone at {} {}".format(N2C(c), x,y))
        return self.applyManualMoves(self.state)

    def undo(self) -> None:
        '''
        external function to undo
        '''
        self._undoLastMove(self.last_x, self.last_y, self.last_color)

    def _undoLastMove(self, x:int, y:int, c_str:str) -> None:
        logging.info('Undo Last Move')
        seq = self.sgf.get_main_sequence()
        if len(seq) > 0:
            last = seq[-2]
            last_move_color = last.properties()
            last_move_pos = last.get(last_move_color[-1])
            if last_move_pos  == (x,y):
                last = seq[-2]
                if len(seq) >=3:
                    self.sgf_node.reparent(last.parent, -1)
                    n = seq[-3].properties()[-1]
                    self.last_color = C2N(n)
                    self.last_x = seq[-3].get_move()[1][0]
                    self.last_y = seq[-3].get_move()[1][1]
                else:
                    self.sgf_node.reparent(self.sgf_node.parent, -1)
                    self.last_color = 2
                    self.last_x = -1
                    self.last_y = -1

        # clear state
        self.state[x,y] = 2
    

    def _setStone(self, x:int, y:int, c_str:str) -> None:
        c = C2N(c_str)
        self.sgf_node.set_move(c_str.lower(),(x,y))
        self.sgf_node = self.sgf.extend_main_sequence()
        self.last_x = x
        self.last_y = y
        logging.info('{}: {}-{}'.format(c_str, x+1, y+1))
        # upon faulty state we would simply copy it ...
        # TODO: implement capute check and ko check...

        self.state[x,y] = c
        self.last_color = c

    def _captureStone(self, x: int, y: int) -> None:
        logging.info('Capture: {}-{}'.format(x+1, y+1))
        self.state[x,y] = 2

    def __getNeighbours(self, state: GoBoardClassification, 
                            x:int, 
                            y:int, 
                            visited: List[Tuple[int,int]]) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
        dx = [-1, 1, 0, 0]
        dy = [ 0, 0,-1, 1]
        other_nbr = []
        same_nbr = []
        cur_color = state[x,y]
        for _dx, _dy in zip(dx,dy):
            __x = max(min(x+_dx, 0), self.board_size)
            __y = max(min(y+_dy, 0), self.board_size)
            if state[__x, __y] != 2 and (__x, __y) not in visited:
                if state[__x,__y] == cur_color:
                    same_nbr.append((__x, __y))
                else:
                    other_nbr.append((__x, __y))
        return (other_nbr, same_nbr)

    def __countLiberties(self, state: GoBoardClassification):
        # make inv binary free fields are 1 occupied are 0
        B = np.zeros_like(state)
        W = np.zeros_like(state)

        B[state==COLOR.BLACK.value] = 1
        W[state==COLOR.WHITE.value] = 1
        #pdb.set_trace()
        _, markersB = cv2.connectedComponents(toByteImage(B), connectivity=4)
        _, markersW = cv2.connectedComponents(toByteImage(W), connectivity=4)
        idsB = np.unique(markersB)
        idsW = np.unique(markersW)
        # remove 0 group
        idsB = np.delete(idsB, [0])
        idsW = np.delete(idsW, [0])
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        #split maps into each component

        def count_groups(ref, other, ids): 
            count = []
            for i in ids:
                groupBinary = ref.copy()
                groupBinary[ref!=i] = 0
                groupBinary[ref==i] = 1

                map = cv2.dilate(toByteImage(groupBinary), kernel) / 255
                # remove core only leave border
                map[groupBinary==1] = 0
                # subtract other colour
                map[other==1] = 0
                count.append(np.sum(map))
            return count
        count_b = count_groups(markersB, W, idsB)
        count_w = count_groups(markersW, B, idsW)
        markersB -= 1
        markersW -= 1
        return markersB, markersW, count_b, count_w

    def _check_move_validity(self, newState: GoBoardClassification) -> GoBoardClassification:
        '''
            We expect that the user is able to undo his last move without manual interaction 
            with the user inface (back buttons will be provided). 
            We reduce the ammount of false positive by checking for previously placed stones
            and do not allow their removal until they are are captured.

        '''

        changes = []
        isInTree, notInTree = self.whichMovesAreInTheGameTree(newState)
        old_markers_b, old_markers_w, count_b, count_w = self.__countLiberties(self.state)

        for (c_str, (x,y)) in isInTree:
            # when the stone was removed check the previous state
            if c_str == 'B':
                id = old_markers_b[x,y]
                liberties = count_b[id]
            elif c_str =='W':
                id = old_markers_w[x,y]
                liberties = count_w[id]
             
            #if liberties != 0:
            #    # something bad happened (detection lost between two steps)
            #    # remove this move revert to old state
            #    logging.warning("Reset board state")
            #    newState[x,y] = self.state[x,y]

        
        # options 
        # a) the same stone was moved -> correction
        # b) one stone was placed either 
        # b-a) capturing n other stones
        # b-b) capturing no other stones but m other stones failed being detected
        # b-c) a mixture of b-a) and b-b)
            
        added_next_color = [x for x in notInTree if x[0] not in ['E', N2C(self.last_color)]]
        added_old_color = [x for x in notInTree if x[0] == N2C(self.last_color)]
        removed = [x for x in notInTree if x[0] == 'E']


        if len(added_old_color) == 1 and len(removed) == 1 and len(added_next_color) == 0: 
            added = added_old_color[0]
            removed = removed[0]
            if (removed[1][0] == self.last_x and removed[1][1] == self.last_y):
                # case a)
                return newState

        elif len(added_next_color) == 1 and len(added_old_color) == 0:
            # apply adding of stone and check liberties
            tempState = self.state.copy()
            c_str, (x,y) = added_next_color[0]
            tempState[x,y] = C2N(c_str)
            old_markers_b, old_markers_w, count_b, count_w = self.__countLiberties(tempState)

            # check that we have not forgotten to remove a stone neighbouring the new placed one
            markers, count = (old_markers_b, count_b) if added_next_color[0] == 'B' else (old_markers_w, count_w)
            for groupId, cnt in enumerate(count): # zip(range(1, len(count)+1) ,count):
                if cnt == 0:
                    #remove group
                    idx = np.argwhere(markers == groupId)
                    for x,y in idx:
                        newState[x,y] = COLOR.NONE.value

            if len(removed) > 0:
                #check all removed stones for validity
                markers, count = (old_markers_b, count_b) if removed[0] == 'B' else (old_markers_w, count_w)
                for _, (rx,ry) in removed:
                    if markers[rx,ry] > -1:
                        if count[markers[rx,ry]] == 0:
                            # case b-a)
                            pass
                        else:
                        # count[markers[rx, ry]] > 0:
                            # case b-b) -> undo removal of stone
                            newState[rx, ry] = self.state[rx, ry]

        return newState


 