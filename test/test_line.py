import unittest
from pygo.utils.line import *

class LineTests(unittest.TestCase):
    def assertNPEqual(self, x, y):
        return self.assertIsNone(np.testing.assert_array_equal(x,y))
    
    def testLineMerge(self):
        line1 = np.array([1,1, 10, 10])
        line2 = np.array([15,15, 12, 12])
        merged1 = merge_lines(line1, line2)

        line1 = np.array([1,1, 10, 10])
        line2 = np.array([12,12, 15, 15])
        merged2 = merge_lines(line1, line2)

        line1 = np.array([10,10, 1, 1])
        line2 = np.array([12,12, 15, 15])
        merged3 = merge_lines(line1, line2)

        line1 = np.array([10,10, 1, 1])
        line2 = np.array([15,15, 12, 12])
        merged4 = merge_lines(line1, line2)



        self.assertNPEqual(merged1, np.array([1,1,15,15]))
        self.assertNPEqual(merged2, np.array([1,1,15,15]))
        self.assertNPEqual(merged3, np.array([1,1,15,15]))
        self.assertNPEqual(merged4, np.array([1,1,15,15]))
if __name__ == '__main__':
    unittest.main()