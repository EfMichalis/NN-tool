import unittest
from nntool.Trainer import *
from nntool.datasets import mnist

class Test_simple(unittest.TestCase):
    def test_trainer_init(self):
        self.failUnlessRaises(Exception, Trainer('top1.txt'))

    
    def test_trainer_train(self):
            trainer = Trainer('top1.txt')
            Train_Examples, Train_Labels, Test_Examples, Test_Labels, Set_Names =mnist()
            
            trainer.SetData(Train_Examples, Train_Labels, Test_Examples, Test_Labels, Set_Names)
            trainer.SetSession()
            
            self.failUnlessRaises(Exception,  trainer.TRAIN(2, 120, Tb=40, Te=1, test_predict=True))
           



    # def test_k2(self):
    #      self.assertEquals(2,2)


if __name__ == '__main__':
    unittest.main()