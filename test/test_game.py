import unittest

from pygo.Game import Game
from pygo.Signals import GameTreeBack, GameTreeForward, Signals
from pygo.utils.color import  N2C, C2N, COLOR
import pdb

from pudb import set_trace


class GameTests(unittest.TestCase):
    def testMultiCaptureWithoutDetection(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(0,1))
        self.game._test_set('W',(0,0))
        self.game._test_set('B',(1,1))
        self.game._test_set('W',(1,0))
        nextState = self.game.state.copy()
        nextState[2,0] = C2N('B')
        nextState[0,0] = C2N('E')
        updatedState = self.game._simple_move_validity_check(nextState)

        self.assertEqual(updatedState[0,0], COLOR.NONE.value)
        self.assertEqual(updatedState[1,0], COLOR.NONE.value)



    def testUndoRedo(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(3,3))
        self.game._test_set('W',(15,15))
        self.game._test_set('B',(3,4))
        self.game._test_set('W',(3,15))

        self.game._Game__game_tree_back()
        self.assertEqual(self.game.last_color, COLOR.BLACK.value)
        self.assertEqual(self.game.last_x, 3)
        self.assertEqual(self.game.last_y, 4)

        self.game._Game__game_tree_back()
        self.assertEqual(self.game.last_color, COLOR.WHITE.value)
        self.assertEqual(self.game.last_x, 15)
        self.assertEqual(self.game.last_y, 15)
 
        self.game._Game__game_tree_forward()
        self.assertEqual(self.game.last_color, COLOR.BLACK.value)
        self.assertEqual(self.game.last_x, 3)
        self.assertEqual(self.game.last_y, 4)

        self.game._Game__game_tree_back()
        self.assertEqual(self.game.last_color, COLOR.WHITE.value)
        self.assertEqual(self.game.last_x, 15)
        self.assertEqual(self.game.last_y, 15)
 
        self.game._Game__game_tree_back()
        self.assertEqual(self.game.last_color, COLOR.BLACK.value)
        self.assertEqual(self.game.last_x, 3)
        self.assertEqual(self.game.last_y, 3)



        
    def testCountLibertiesCorner(self):
        self.game = Game()
        self.game.state[0,0] = COLOR.BLACK.value
        self.game.state[18,0]  = COLOR.BLACK.value
        self.game.state[18,18] = COLOR.WHITE.value
        self.game.state[0,18] = COLOR.WHITE.value
        mb, mw, cb, cw = self.game._Game__countLiberties(self.game.state)

        self.assertEqual(cb, [2,2])
        self.assertEqual(cw, [2,2])
        self.assertEqual(cb[mb[0,0]], 2)
        self.assertEqual(cb[mb[18,0]], 2)
        self.assertEqual(cw[mw[18,18]], 2)
        self.assertEqual(cw[mw[0, 18]], 2)

    def testCountLibertiesCornerBorderingStones(self):
        self.game = Game()
        self.game.state[0,0] = COLOR.BLACK.value
        self.game.state[0,1]  = COLOR.BLACK.value
        self.game.state[1,0] = COLOR.WHITE.value
        self.game.state[1,1] = COLOR.WHITE.value
        mb, mw, cb, cw = self.game._Game__countLiberties(self.game.state)
        self.assertEqual(cb, [1])
        self.assertEqual(cw, [3])
        self.assertEqual(cb[mb[0,0]], 1)
        self.assertEqual(cw[mw[1,0]], 3)

    def testCaptureWithDetection(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(0,1))
        self.game._test_set('W',(0,0))
        nextState = self.game.state.copy()
        nextState[1,0] = C2N('B')
        nextState[0,0] = C2N('E')

        updatedState = self.game._simple_move_validity_check(nextState)

        self.assertEqual(updatedState[0,0], COLOR.NONE.value)
        self.assertEqual(updatedState[1,0], COLOR.BLACK.value)

    def testCaptureWithDetectionNoBorder(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(3,3))
        self.game._test_set('W',(3,4))
        self.game._test_set('B',(4,4))
        self.game._test_set('W',(13,4))
        self.game._test_set('B',(2,4))
        self.game._test_set('W',(1,1))
        #self.game._test_set('B',(3,5))

        nextState = self.game.state.copy()
        nextState[3,4] = C2N('E')
        nextState[3,5] = C2N('B')

        updatedState = self.game._simple_move_validity_check(nextState)

        self.assertEqual(updatedState[3,4], COLOR.NONE.value)
        self.assertEqual(updatedState[3,5], COLOR.BLACK.value)

    def testCaptureWithDetectionNoBorderPartialDetection(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(3,3))
        self.game._test_set('W',(3,4))
        self.game._test_set('B',(4,4))
        self.game._test_set('W',(13,4))
        self.game._test_set('B',(2,4))
        self.game._test_set('W',(1,1))
        #self.game._test_set('B',(3,5))

        nextState = self.game.state.copy()
        #nextState[3,4] = C2N('E')
        nextState[3,5] = C2N('B')

        updatedState = self.game._simple_move_validity_check(nextState)

        self.assertEqual(updatedState[3,4], COLOR.NONE.value)
        self.assertEqual(updatedState[3,5], COLOR.BLACK.value)

    def testCaptureWithDetectionNoBorderFalseDetection(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(3,3))
        self.game._test_set('W',(3,4))
        self.game._test_set('B',(4,4))
        self.game._test_set('W',(13,4))
        self.game._test_set('B',(2,4))
        self.game._test_set('W',(1,1))
        #self.game._test_set('B',(3,5))

        nextState = self.game.state.copy()
        nextState[3,4] = C2N('W')
        nextState[3,5] = C2N('B')

        updatedState = self.game._simple_move_validity_check(nextState)

        self.assertEqual(updatedState[3,4], COLOR.NONE.value)
        self.assertEqual(updatedState[3,5], COLOR.BLACK.value)

    def testCaptureWithDetectionNoBorderFalseDetectionDoubleAtari(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(4,4))
        self.game._test_set('W',(3,4))
        self.game._test_set('B',(3,5))
        self.game._test_set('W',(4,5))
        self.game._test_set('B',(4,6))
        self.game._test_set('W',(3,6))

        nextState = self.game.state.copy()
        nextState[4,5] = C2N('W')
        nextState[5,5] = C2N('B')

        updatedState = self.game._simple_move_validity_check(nextState)

        self.assertEqual(updatedState[4,5], COLOR.NONE.value)
        self.assertEqual(updatedState[5,5], COLOR.BLACK.value)

    def testCaptureWithDetectionNoBorderCorrectDetectionDoubleAtari(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(4,4))
        self.game._test_set('W',(3,4))
        self.game._test_set('B',(3,5))
        self.game._test_set('W',(4,5))
        self.game._test_set('B',(4,6))
        self.game._test_set('W',(3,6))

        nextState = self.game.state.copy()
        nextState[4,5] = C2N('E')
        nextState[5,5] = C2N('B')

        updatedState = self.game._simple_move_validity_check(nextState)

        self.assertEqual(updatedState[4,5], COLOR.NONE.value)
        self.assertEqual(updatedState[5,5], COLOR.BLACK.value)


    def testCaptureWithDetectionNoBorderPartialCorrectDetectionDoubleAtari(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(4,4))
        self.game._test_set('W',(3,4))
        self.game._test_set('B',(3,5))
        self.game._test_set('W',(4,5))
        self.game._test_set('B',(4,6))
        self.game._test_set('W',(3,6))

        nextState = self.game.state.copy()
        #nextState[3,4] = C2N('N')
        nextState[5,5] = C2N('B')
        updatedState = self.game._simple_move_validity_check(nextState)

        self.assertEqual(updatedState[4,5], COLOR.NONE.value)
        self.assertEqual(updatedState[5,5], COLOR.BLACK.value)




    def testMultiCaptureWithDetection(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(0,1))
        self.game._test_set('W',(0,0))
        self.game._test_set('B',(1,1))
        self.game._test_set('W',(1,0))
        self.game._test_set('B',(2,1))
        self.game._test_set('W',(2,0))
        nextState = self.game.state.copy()
        nextState[3,0] = C2N('B')
        nextState[0,0] = C2N('E')
        nextState[1,0] = C2N('E')
        nextState[2,0] = C2N('E')

        updatedState = self.game._simple_move_validity_check(nextState)

        self.assertEqual(updatedState[0,0], COLOR.NONE.value)
        self.assertEqual(updatedState[1,0], COLOR.NONE.value)


    def testCaptureWithoutDetection(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(0,1))
        self.game._test_set('W',(0,0))
        nextState = self.game.state.copy()
        nextState[1,0] = C2N('B')

        updatedState = self.game._simple_move_validity_check(nextState)
        
        self.assertEqual(updatedState[0,0], COLOR.NONE.value)
        self.assertEqual(updatedState[1,0], COLOR.BLACK.value)



if __name__ == '__main__':
    unittest.main()