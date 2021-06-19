import unittest
from nntool.Trainer import *
from nntool.datasets import *

class TestTrainer(unittest.TestCase):
    def setUp(self):
        tf.compat.v1.disable_eager_execution()
        self.failUnlessRaises(Exception, Trainer('top.txt'))

    
    def test_trainer_train(self):
        tf.compat.v1.disable_eager_execution()
        trainer = Trainer('top.txt')
        dsets = datasets()
        Train_Examples, Train_Labels, Test_Examples, Test_Labels, Set_Names = dsets.mnist()

        trainer.SetData(Train_Examples, Train_Labels, Test_Examples, Test_Labels, Set_Names)
        trainer.SetSession()
            
        self.failUnlessRaises(Exception,  trainer.TRAIN(2, 120, Tb=40, Te=1, test_predict=True))


if __name__ == '__main__':
    unittest.main()
