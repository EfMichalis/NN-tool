import unittest


class Test_simple(unittest.TestCase):
    def test_k1(self):
        self.assertEquals(3,3)
    def test_k2(self):
         self.assertEquals(2,2)


if __name__ == '__main__':
    unittest.main()