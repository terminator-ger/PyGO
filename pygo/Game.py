import cv2
import logging

import random
import numpy as np

from enum import Enum, auto
from datetime import datetime
from playsound import playsound

from sgfmill import sgf, boards, sgf_moves
from typing import List, TextIO, Tuple, Optional

from pygo.utils.image import toByteImage
from pygo.utils.color import N2C, C2N, COLOR, CNOT
from pygo.utils.debug import DebugInfo, DebugInfoProvider, Timing
from pygo.utils.typing import Move, NetMove, GoBoardClassification
from pygo.utils.misc import coordinate_to_letter, pygo_to_go_coord_sys, sgfmill_to_pygo_coord_sys
from pygo.Signals import *


class GameState(Enum):
    RUNNING = auto()
    PAUSED = auto()
    NOT_STARTED = auto()

class MoveValidationAlg(Enum):
    NONE = auto()
    ONE_MOVE = auto()
    TWO_MOVES = auto()
    MULTI_MOVES = auto()


class Game(DebugInfoProvider, Timing):
    def __init__(self):
        DebugInfoProvider.__init__(self)
        Timing.__init__(self)

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
        self.game_tree = None
        self.manualMoves = []
        self.state_history = []
        self.cur_time = None

        self.settings  = {
            'AllowUndo': False,
            'MoveValidation': MoveValidationAlg.TWO_MOVES
        }

        Signals.subscribe(OnSettingsChanged, self.settings_updated)
        Signals.subscribe(GameRun, self.__game_run)
        Signals.subscribe(GamePause, self.__game_pause)
        Signals.subscribe(GameReset, self.__resetGameState)

        Signals.subscribe(GameNew, self.__startNewGame)
        Signals.subscribe(GameTreeBack, self.__game_tree_back)
        Signals.subscribe(GameTreeForward, self.__game_tree_forward)

    #######################################
    # Temporal Manipulation Routines
    #######################################

    def __game_tree_back(self, *args):
        if self.GS == GameState.NOT_STARTED:
            return

        if self.GS == GameState.RUNNING:
            self.GS = GameState.PAUSED
            logging.info("Paused Game")

        last = self.game_tree.get_last_node().parent

        if last.get_move() != (None, None):
            n, (mx,my) = last.get_move()
            x,y = sgfmill_to_pygo_coord_sys((mx,my))
            self.state[x,y] = C2N('E')
            
            if last.parent.parent is not None:
                n, (mx,my) = last.parent.get_move()
                x,y = sgfmill_to_pygo_coord_sys((mx,my))
            else:
                n = 'E'
                x = -1
                y = -1

            if last.parent is not None:
                last.reparent(last.parent, 1)
                last.parent.new_child(0)
 
            self.last_color = C2N(n)
            self.last_x = x
            self.last_y = y

            Signals.emit(UpdateLog, self.get_move_list())


    def __game_tree_forward(self, *args):
        if self.GS == GameState.NOT_STARTED:
            return

        if self.GS == GameState.RUNNING:
            self.GS = GameState.PAUSED
            logging.info("Paused Game")

        par_node = self.game_tree.get_last_node().parent
        
        # moving back in time removes moves from the leftmost variation
        if len(par_node) > 1:
            next_moves = par_node[1]
            dead_end = par_node[0]
            dead_end.delete()

            # delete the other node as we would generate multiple 
            # dead ends skipping fwd and bwd
            n, (mx,my) = next_moves.get_move()
            x,y = sgfmill_to_pygo_coord_sys((mx,my))

            self.state[x,y] = C2N(n)
            self.last_color = C2N(n)
            self.last_x = x
            self.last_y = y
            next_moves.reparent(par_node, 0)
            Signals.emit(UpdateLog, self.get_move_list())


    def isPaused(self):
        return True if self.GS == GameState.PAUSED else False


    def __startNewGame(self, *args) -> None:
        self.startNewGame()


    def __game_pause(self, *args):
        logging.info("Paused Game")
        self.GS = GameState.PAUSED


    def __game_run(self, *args):
        logging.info("Resume Game")
        self.GS = GameState.RUNNING


    def __resetGameState(self, *args) -> None:
        logging.debug("GAME: Reset Game state")
        size = 19
        self.game_tree = sgf.Sgf_game(size=size)
        self.game_tree.extend_main_sequence()
        self.GS = GameState.NOT_STARTED
        self.state = np.ones((self.board_size, self.board_size), dtype=int)*2
        self.board_size = size
        self.last_x = -1
        self.last_y = -1
        self.last_color = 2
        self.manualMoves = []


    def startNewGame(self, size=19) -> None:
        logging.debug("GAME: Starting new Game")
        self.__resetGameState()
        self.cur_time = datetime.now().strftime("%d-%m-%Y")

        self.GS = GameState.RUNNING
        Signals.emit(UpdateLog, self.get_move_list())


    def undo(self) -> None:
        '''
        external function to undo
        '''
        self._undoLastMove(self.last_x, self.last_y, self.last_color)


    def _undoLastMove(self, x:int, y:int, c_str:str) -> None:
        if self.GS == GameState.RUNNING:
            self.GS == GameState.PAUSED
            logging.info('Undo Last Move')
        last = self.game_tree.get_last_node()
        if last:
            last.parent.new_child(0)
            n, (x,y) = last.get_move()
            self.state[x,y] = C2N('E')
            if last.parent:
                n, (x,y) = last.parent.get_move()
            else:
                n = 'E'
                x = -1
                y = -1
            self.last_color = C2N(n)
            self.last_x = x
            self.last_y = y


    #######################################
    # Manual Manipulation Routines
    #######################################

    def setManual(self, x:int ,y:int, color:Optional[int]=None) -> None:
        ''' 
            Manual override for single moves
            Fallback in case some detection does not work als intended
        '''
        # either add or remove stone
        if color == 2 or self.state[x,y] != 2:
            # delete existing move
            logging.debug("Remove Stone at {} {}".format(x,y))
            if x == self.last_x and y == self.last_y:
                # last move was probably a false detection
                # reset last color
                self._undoLastMove(x,y,None)
            
            #clear move
            self.manualMoves.append([2, (x,y)])
        elif color in [0,1]:
            c = color
            self._setStone(x, y, N2C(c))
            #save move in overwrite state
            self.manualMoves.append([c, (x,y)])
            logging.debug("Manual overwrite: Adding {} Stone at {} {}".format(N2C(c), x+1,y+1))
        else:
            # add stone
            c = self.nextMove()
            self._setStone(x, y, N2C(c))
            #save move in overwrite state
            self.manualMoves.append([c, (x,y)])
            logging.debug("Manual overwrite: Adding {} Stone at {} {}".format(N2C(c), x+1,y+1))

        Signals.emit(UpdateLog, self.get_move_list())
        return self.applyManualMoves(self.state)


    def clearManual(self, x:int, y:int) -> None:
        del_move = None
        for c,(x_,y_) in self.manualMoves:
            if x_ == x and y_ == y:
                del_move = [c, (x_,y_)]
        if del_move is not None:
            self.manualMoves.remove(del_move)


    def clearManualAll(self) -> None:
        self.manualMoves = []


    def applyManualMoves(self, state: GoBoardClassification ) -> GoBoardClassification:
        for (c, (x,y)) in self.manualMoves:
            state[x,y] = c
        return state



    #######################################
    # Evaluation of moves
    #######################################

    def updateStateNoChecks(self, state) -> NetMove:
        state = state.reshape(19,19)
        idx = np.argwhere(np.abs(self.state-state)>0)
        if len(idx) > 0:
            x = idx[0,0]
            y = idx[0,1]
            c = state[x,y]
            rnd = random.randint(1,5) 
            playsound('sounds/stone{}.wav'.format(rnd), block=False)
            c_str = N2C(c)

            print('{}: {}-{}'.format(c_str, x+1, y+1))
            self.last_x = x
            self.last_y = y
            self.state = state
            self.last_color = c
            return [c_str.lower(), x, y]


    def _setStone(self, x:int, y:int, c_str:str) -> None:
        c = C2N(c_str)
        
        # sgf mill accepts points as (row, col) with 
        # a coordinate systems used on go board not in sgf (inverted y-axis!)

        mill_x, mill_y = (self.board_size-y), x

        #import pdb
        #pdb.set_trace()
        self.game_tree.get_last_node().set_move(c_str.lower(),(mill_x, mill_y))
        self.game_tree.extend_main_sequence()

        x_, y_ = pygo_to_go_coord_sys((x,y))
        logging.info('{}: {}-{}'.format(c_str, x_, y_))
        self.last_x = x
        self.last_y = y
        self.state[x,y] = c
        self.last_color = c
        if self.GS == GameState.RUNNING:
            Signals.emit(UpdateLog, self.get_move_list())


    def _captureStone(self, x: int, y: int) -> None:
        x_, y_ = pygo_to_go_coord_sys((x,y))
        logging.info('Capture: {}-{}'.format(x_, y_))
        self.state[x,y] = 2


    def __countLiberties(self, state: GoBoardClassification):
        # make inv binary free fields are 1 occupied are 0
        B = np.zeros_like(state)
        W = np.zeros_like(state)

        B[state==COLOR.BLACK.value] = 1
        W[state==COLOR.WHITE.value] = 1
        
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


    def _simple_move_validity_check(self, newState: GoBoardClassification) -> GoBoardClassification:
        '''
            We reduce the ammount of false positive by checking for previously placed stones
            and do not allow their removal until they are are captured.

        '''
        isInTree, notInTree = self.whichMovesAreInTheGameTree(newState)
        
        added_next_color = [x for x in notInTree if x[0] not in ['E', N2C(self.last_color)]]
        added_old_color = [x for x in notInTree if x[0] == N2C(self.last_color)]
        removed_new_color = [x for x in notInTree if x[0] == 'E' and self.state[x[1][0],x[1][1]] == CNOT(self.last_color)]
        
        # apply add stone to temp state
        working_state = self.state.copy()
        if len(added_next_color) == 1:
            c_str, (x,y) = added_next_color[0]
            working_state[x,y] = C2N(c_str)
        else:
            return self.state
        
        old_markers_b, old_markers_w, count_b, count_w = self.__countLiberties(working_state)
        # after stone was added check for removed of old color
        if self.last_color == C2N('B'):
            count = count_b 
            markers = old_markers_b
        elif self.last_color == C2N('W'):
            count = count_w
            markers = old_markers_w
        else:
            # start of new game
            count = []
            markers = []

        for groupId, cnt in enumerate(count): 
            if cnt == 0:
                #remove group
                idx = np.argwhere(markers == groupId)
                for x,y in idx:
                    logging.debug('Removing {}-{} {} from the manual moves'.format(c_str, x, y))
                    working_state[x,y] = COLOR.NONE.value

        for (c_str, (x,y)) in removed_new_color:
            # cannot happen
            working_state[x,y] = self.state[x,y]
        
        for (c_str, (x,y)) in added_old_color:
            # cannot happen
            working_state[x,y] = self.state[x,y]

        for (c_str, (x,y)) in isInTree:
            # cannot happen
            working_state[x,y] = self.state[x,y]

        return working_state


    def updateState(self, state) -> None:
        '''
        input detected state from classifier
        '''
        state = self.applyManualMoves(self._unravel(state))
        if self.GS == GameState.RUNNING:
            self.updateStateWithChecks(state)

            if self.settings['MoveValidation'] == MoveValidationAlg.TWO_MOVES:
                # run a second time in case we have two moves 
                self.updateStateWithChecks(state)
   

        elif self.GS == GameState.PAUSED:
            return
        elif self.GS == GameState.NOT_STARTED:
            self.updateStateNoChecks(state)
        else:
            logging.error("Unkown Game State")
            raise RuntimeError("Unkonwn Game State")


    def whichMovesAreInTheGameTree(self, state: GoBoardClassification) -> Tuple[List[Move],List[Move]]:
        '''
            Splits the classifiers state into moves which are allready in the game tree
            (played and stored) and new moves
        '''
        idx = np.argwhere(np.abs(self.state-state)>0)
        X = idx[:,0]
        Y = idx[:,1]
        isInTree = []
        notInTree = []
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            c = state[x,y]
            c_str = N2C(c)
            seq = self.game_tree.get_main_sequence()[:-2]
            for node in seq:
                if node.properties()[-1] == c_str and (x,y) == node.get(c_str):
                    isInTree.append((c_str, (x,y)))
                    break
            # not found
            notInTree.append((c_str,(x,y)))
        return (isInTree, notInTree)


    def _check_handicap(self, newState) -> NetMove:
        newState = newState.reshape(19,19)
        isInTree, notInTree = self.whichMovesAreInTheGameTree(newState)
        # we just initialized a new game we could have handicap stones
        # all black and on the defined positions
        handicap_positions = [(3,3), (3,15), (15,3), (15,15), 
                              (3,9),  (9,3), (15,9), (9, 15), 
                              (9,9)]
        BoardSetup = boards.Board(self.board_size)

        # we need black stones, more than one and only on start points 
        if (all([x[0]=='B' for x in notInTree]) 
            and all([x[1] in handicap_positions for x in notInTree])
            and len([x[0]=='B' for x in notInTree]) > 1):

            moves_black = []
            moves_white = []
            for c_str, (x,y) in notInTree:
                if c_str == "B":
                    self.state[x,y] = C2N("B")

                    mill_x, mill_y = (self.board_size-y), x
                    moves_black.append((mill_x,mill_y))

            BoardSetup.apply_setup(moves_black, moves_white, [])
            sgf_moves.set_initial_position(self.game_tree, BoardSetup)
            self.game_tree.get_main_sequence()[0].set("HA", len(moves_black))
            
            # White starts
            self.last_color = C2N("B")
            # update log
            Signals.emit(UpdateLog, self.get_move_list())
            Signals.emit(GameHandicapMove, len(moves_black)+len(moves_white))
        else:
            logging.info('No handicap detected')
            # start with black
            self.last_color = C2N('W')

    
    def updateStateWithChecks(self, state) -> bool:
        if self.last_color == 2:
            self._check_handicap(state)
        
        if np.array_equal(state, self.state):
            logging.debug('No Difference found')
            return
        logging.debug2('before validity check') 
        logging.debug2(state.reshape(19,19))
        state = self._simple_move_validity_check(state)
        logging.debug2('after validity check') 
        logging.debug2(state.reshape(19,19))

        isInTree, notInTree = self.whichMovesAreInTheGameTree(state)
        
        for (c_str, (x,y)) in notInTree:
            if c_str !='E':
                logging.debug('Adding {} at {} {}'.format(c_str, x, y))
                self._setStone(x, y, c_str)
                Signals.emit(UpdateHistory, state, True, c_str)

            if c_str == 'E':
                logging.debug('Removing {} at {} {}'.format(c_str, x, y))
                self._captureStone(x, y)
                Signals.emit(UpdateHistory, state, False, c_str)




    #######################################
    # Miscellaneous
    #######################################

    def print(self):
        root_tree = self.game_tree.get_main_sequence()
        for node in root_tree:
            print(node.get_move())


    def settings_updated(self, args):
        new_settings = args[0]
        for k in self.settings.keys():
            if k in new_settings.keys():
                self.settings[k] = new_settings[k].get()


    def getCurrentState(self) -> GoBoardClassification:
        return self.state


    def saveGame(self, file: TextIO = None) -> None:
        if file is None:
            cur_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
            file = '{}.sgf'.format(cur_time)
            with open(file, 'wb') as f:
                f.write(self.game_tree.serialise())
        else:
            file.write(self.game_tree.serialise())


    def _test_set(self, stone, coords) -> None:
        x = coords[0]
        y = coords[1]
        c_str = stone

        mill_x, mill_y = (self.board_size-y), x

        self.game_tree.get_last_node().set_move(c_str.lower(),(mill_x,mill_y))
        self.game_tree.extend_main_sequence()
        self.last_x = x
        self.last_y = y

        x_, y_ = pygo_to_go_coord_sys((x,y))
        logging.info('{}: {}-{}'.format(c_str, x_, y_))
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
            return 0


    def _unravel(self, x):
        return x.reshape(self.board_size, self.board_size)


    def get_move_list(self) -> List[str]:
        moves = []
        seq = self.game_tree.get_main_sequence()
        handicap = self.game_tree.get_handicap()

        if self.cur_time is not None:
            moves.append('{}\n'.format(self.cur_time))
            moves.append("===============")
        
        if handicap is not None and handicap > 1:
            moves.append("Handicap +{}\n".format(handicap))
        
        for node in seq:
            if node is not None:
                mv = node.get_move()
                if mv != (None, None):
                    n, (x,y) = mv

                    moves.append("{}: {}-{}\n".format(n.upper(), 
                                                        coordinate_to_letter(y),
                                                        x))

        return moves

 




















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







    def _move_validity_check(self, newState: GoBoardClassification) -> GoBoardClassification:
        '''
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
                if self.settings['AllowUndo']:
                    return newState
                else:
                    return self.state


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
        else:
            # check removed
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
            for n, (x,y) in added_old_color:
                newState[x,y] = C2N("E")
        return newState

