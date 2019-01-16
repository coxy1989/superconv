import unittest
from modules import callbacks as cbs

class TestSchedulers(unittest.TestCase):

    def test_1cycle_mom(self):
        actual = cbs._1cycle_mom(0, 100, 0, 1)
        expected = 1
        self.assertEqual(actual, expected)

        actual = cbs._1cycle_mom(49, 100, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = cbs._1cycle_mom(100, 100, 0, 1)
        expected = 1
        self.assertEqual(actual, expected)

        actual = cbs._1cycle_mom(150, 100, 0, 1)
        expected = 1
        self.assertEqual(actual, expected)

    def test_1cycle_lr(self):

        actual = cbs._1cycle_lr(0, 101, 10, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = cbs._1cycle_lr(50, 101, 10, 0, 1)
        expected = 1
        self.assertEqual(actual, expected)

        actual = cbs._1cycle_lr(100, 101, 10, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)

        actual = cbs._1cycle_lr(150, 101, 10, 0, 1)
        expected = 0
        self.assertEqual(actual, expected)
