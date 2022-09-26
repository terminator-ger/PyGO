import unittest

from pygo.Game import Game
from pygo.utils.color import  N2C, C2N, COLOR
import pdb


class GameTests(unittest.TestCase):
        
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

        updatedState = self.game._check_move_validity(nextState)

        self.assertEqual(updatedState[0,0], COLOR.NONE.value)
        self.assertEqual(updatedState[1,0], COLOR.BLACK.value)

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

        updatedState = self.game._check_move_validity(nextState)

        self.assertEqual(updatedState[0,0], COLOR.NONE.value)
        self.assertEqual(updatedState[1,0], COLOR.NONE.value)


    def testCaptureWithoutDetection(self):
        self.game = Game()
        self.game.startNewGame()
        self.game._test_set('B',(0,1))
        self.game._test_set('W',(0,0))
        nextState = self.game.state.copy()
        nextState[1,0] = C2N('B')

        updatedState = self.game._check_move_validity(nextState)
        
        self.assertEqual(updatedState[0,0], COLOR.NONE.value)
        self.assertEqual(updatedState[1,0], COLOR.BLACK.value)


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

        updatedState = self.game._check_move_validity(nextState)

        self.assertEqual(updatedState[0,0], COLOR.NONE.value)
        self.assertEqual(updatedState[1,0], COLOR.NONE.value)

if __name__ == '__main__':
    unittest.main()