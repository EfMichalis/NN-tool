from struct import *
import matplotlib.pyplot as plt
import os
import sys
from pandas import DataFrame as DF
import csv
import tensorflow as tf
import time
from inspect import stack as InStack
import numpy as np


class Trainer:
    def __init__(self, topology_file, weights_path=None):
        self.prdir = None
        dirq = InStack()[1][1][:-6]
        self.buildnet(topology_file, weights_path)

    def buildnet(self, topology_file, weights_path=None):
            self.topology_file = topology_file
            self.weights_path = weights_path
            self.NetworkNewTrainig = True  # if True -> set default parameters in optimization algorithm (e.g learnig rate)
            if weights_path is None:
                print('Not Find Weight and Bias .Bin files.')
                print('New Weight and Bias .Bin files.')
            else:
                if self.weights_path is None:
                    print('Weight and Bias .Bin allready exist')
                    print('Do you want to overwright?\nNo : 0\nYes : 1')
                    Ans = input()
                    while Ans != "0" and Ans != "1":
                        print('No : 0\nYes : 1')
                        Ans = input()
                    if Ans == "0":
                        pass

            if self.weights_path is not None:
                # Open Weights Binary Files
                try:
                    self.Weights_Bin_File = open('{}/weights.bin'.format(self.weights_path), 'br')
                    self.Bias_Bin_File = open('{}/bias.bin'.format(self.weights_path), 'br')
                    self.ScaleFile = open('{}/scale.bin'.format(self.weights_path), 'br')
                    self.BetaFile = open('{}/beta.bin'.format(self.weights_path), 'br')
                except Exception as e:
                    pass
            if os.system('ls '+self.topology_file) == 512:
                sys.exit("Not find topology.txt file")
            self.readfile()
            if self.weights_path is not None:
                self.Weights_Bin_File.close()
                self.Bias_Bin_File.close()

    def readfile(self):
        Str = ''
        with open(self.topology_file, "r") as fl:
            Str = fl.read().replace('\n', '').split(';')  # Split topology.txt by ';' in a list
            Str = Str[:-1]
            self.Topology = []  # List with sublist of  Layer's layer type and layer parameters info
            self.Idropout = 0  # keeps from self.Topology the previous Layer's index from the last Dropout layer
            self.DroupoutCnt = 0
            self.DroupoutsProbabilitys = np.array([])  # Storing for trainig processing
            self.keep_prob = tf.compat.v1.placeholder(tf.float32)  # During the trainig processing, the DroupoutsProbabilitys is placed here
            self.Idbatchn = 0 # keeps from self.Topology the previous Layer's index from the last BatchNorm layer
            self.Layers = []  # Storing LayerType methods.
            self.Weights = []  # Storing Weights Per Layer as necessary.
            self.Bias = []  # Same with Weights
            self.Beta = []
            self.Scale = []
            # Starting read topology syndax
            if(Str[0][:Str[0].index('(')] != 'Input'):  # if the  topology.txt does't start with Input Layer
                sys.exit("Topology must start with 'Input' object e.g. 'Input(x,y,z)'")
            else:
                for i in range(0, len(Str)):
                    print(Str[i])
                    self.case(Str[i])

        self.y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, self.LayerShape(self.Layers[-1])[-1]])  # Here placed the true classes ,the Layers[-1] is the classes that were predicted

    def case(self, Stri):
        layertype = Stri[:Stri.index('(')]

        layerparams = Stri[Stri.index('(')+1:-1].split(',')
        # examples layerparams [2,'2','1'],[4],[9,32,'Sigmoid']
        try:
            layerparams[0] = int(layerparams[0])

        except ValueError:
            try:
                layerparams[0] = float(layerparams[0])
            except ValueError:
                layerparams[0] =None
        if layertype == 'Conv':
            layerparams[1] = int(layerparams[1])
            if self.Topology[-1][0] in ['Pool', 'Conv']:
                pass
            elif self.Topology[-1][0] in ['Dropout','BatchNorm']:
                if not(self.Topology[self.Idropout][0] in ['Pool', 'Conv']):
                    sys.exit('the conection of '+layertype+' and '+self.Topology[self.Idropout][0]+' is not valid')

            elif self.Topology[-1][0] == 'Input':
                if not(self.Topology[-1][1][0] >= 2 and self.Topology[-1][1][0] == self.Topology[-1][1][1]):
                    sys.exit('Must Input x,y >=2')
            else:
                sys.exit('the conection of '+layertype+' and '+self.Topology[-1][0]+' is not valid')
            self.HiSize(ConvSize=layerparams[1])
            shape = [layerparams[1], layerparams[1], self.LayerShape(self.Layers[-1])[-1], layerparams[0]]
            self.add_weights(shape)
            # Add Convolutional layer
            self.Layers.append(0)
            self.Layers[-1] = self.conv2d(self.Layers[-2], self.Weights[-1])+self.Bias[-1]

            self.add_layer_with_weight_type(layerparams[-1])

            self.Topology.append([layertype, layerparams])
        elif layertype == 'Pool':
            if self.Topology[-1][0] in ['Pool', 'Conv']:
                pass
            elif self.Topology[-1][0] in ['Dropout','BatchNorm']:
                if not(self.Topology[self.Idropout][0] in ['Pool', 'Conv'] or self.Topology[self.Idbatchn][0] in ['Pool', 'Conv']):
                    sys.exit('Cant Conect '+layertype+" and "+self.Topology[self.Idropout][0])
            elif self.Topology[-1][0] == 'Input':
                if not(self.Topology[-1][1][0] >= 2 and self.Topology[-1][1][0] == self.Topology[-1][1][1]):
                    sys.exit('Must Input x,y >=2')
            else:
                sys.exit('the conection of '+layertype+' and '+self.Topology[-1][0]+' is not valid')
            self.HiSize(PoolSize=layerparams[0])

            # Add Pooling Layer
            self.Layers.append(0)
            if layerparams[1] == 'max':
                self.Layers[-1] = self.max_pool(self.Layers[-2], layerparams[0])
            elif layerparams[1] == 'min':
                self.Layers[-1] = self.min_pool(self.Layers[-2], layerparams[0])
            elif layerparams[1] == 'avg':
                self.Layers[-1] = self.avg_pool(self.Layers[-2], layerparams[0])
            else:
                sys.exit(layerparams[1]+" is't valid pooling method")
            self.Topology.append([layertype, layerparams])

        elif layertype == 'Fc':
            if self.Topology[-1][0] == 'Fc':
                FlatLayer = self.Layers[-1]
            elif self.Topology[-1][0] in ['Dropout','BatchNorm']:
                if self.Topology[self.Idropout][0] in ['Pool', 'Conv'] or self.Topology[self.Idbatchn][0] in ['Pool', 'Conv']:
                    FlatLayer = tf.reshape(self.Layers[-1], [-1, self.multiply(self.LayerShape(self.Layers[-1]))])
                elif self.Topology[self.Idropout][0] == 'Fc' or self.Topology[self.Idbatchn][0] == 'Fc':
                        FlatLayer = self.Layers[-1]
                else:
                    sys.exit('Cant Conect '+layertype+" and "+self.Topology[self.Idropout][0])
            elif self.Topology[-1][0] in ['Pool', 'Conv', 'Input']:
                FlatLayer = tf.reshape(self.Layers[-1], [-1, self.multiply(self.LayerShape(self.Layers[-1]))])
            else:
                sys.exit('the conection of '+layertype+' and '+self.Topology[-1][0]+' is not valid')

            shape = [self.multiply(self.LayerShape(self.Layers[-1])), layerparams[0]]
            self.add_weights(shape)

            self.Layers.append(0)
            self.Layers[-1] = tf.matmul(FlatLayer, self.Weights[-1])+self.Bias[-1]


            self.add_layer_with_weight_type(layerparams[-1])
            self.Topology.append([layertype, layerparams])

        elif layertype == 'Dropout':
            if self.Topology[-1][0] != 'Dropout':
                self.Idropout = len(self.Topology)-1
            # keep_prob=tf.Variable(tf.constant(layerparams[0]/100,dtype='float32'))
            self.Layers[-1] = tf.compat.v1.nn.dropout(self.Layers[-1], tf.gather(self.keep_prob, [self.DroupoutCnt]))
            self.Topology.append([layertype, layerparams])
            self.DroupoutCnt += 1
            self.DroupoutsProbabilitys = np.concatenate([self.DroupoutsProbabilitys, np.array([layerparams[0]])],axis=0)
        
        elif layertype == 'BatchNorm':
            if self.Topology[-1][0] != 'BatchNorm':
                self.Idbatchn = len(self.Topology)-1
            # self.Layers[-1] = tf.layers.batch_normalization(self.Layers[-1])

            shape = self.LayerShape(self.Layers[-1])
            self.add_batch_norm_vars(shape)
            self.Layers.append(0)
            batch_mean, batch_var = tf.nn.moments(self.Layers[-2],list(range(len(self.LayerShape(self.Layers[-2])))))
            self.Layers[-1] = self.batch_normalization(self.Layers[-2], self.Beta[-1], self.Scale[-1],batch_mean,batch_var)
            self.Topology.append([layertype, layerparams])
        elif layertype == 'Input':
            layerparams[1] = int(layerparams[1])
            layerparams[2] = int(layerparams[2])
            self.Layers.append(0)
            self.Layers[-1] = tf.compat.v1.placeholder(tf.float32, shape=[None, layerparams[0], layerparams[1], layerparams[2]])
            self.Topology.append([layertype, layerparams])
            self.Hi = layerparams[0]

        else:
            sys.exit('Not valid Layer Type.')

    # START POOL FUNCTIONS ####################################################
    def max_pool(self, x, poolsize):
        return tf.nn.max_pool2d(x, ksize=[1, poolsize, poolsize, 1],
                              strides=[1, poolsize, poolsize, 1], padding='VALID')

    def avg_pool(self, x, poolsize):
        return tf.nn.avg_pool2d(x, ksize=[1, poolsize, poolsize, 1],
                              strides=[1, poolsize, poolsize, 1], padding='VALID')

    def min_pool(self, x, poolsize):
        return -self.max_pool(-x, poolsize)
    # END POOL FUNCTIONS ######################################################

    # START WEIGHTS AND BIAS FUNCTIONS ########################################
    def add_weights(self, shape):
        self.Weights.append(0)
        self.Bias.append(0)
        if self.weights_path is None:
            self.Weights[-1] = self.new_weight_variable(shape)
            self.Bias[-1] = self.new_bias_variable([shape[-1]])
        else:
            I = self.multiply(shape)
            # Weights
            W = []
            for i in range(0, I):
                W.append(unpack('f', self.Weights_Bin_File.read(4))[0])
            self.Weights[-1] = tf.Variable(tf.reshape(tf.constant(W, dtype='float32'), shape))
            # Bias
            B = []
            for i in range(0, shape[-1]):
                B.append(unpack('f', self.Bias_Bin_File.read(4))[0])
            self.Bias[-1] = tf.Variable(tf.constant(B, dtype='float32'))

    def add_batch_norm_vars(self, shape):
        self.Scale.append(0)
        self.Beta.append(0)
        if self.weights_path is None:
            self.Scale[-1], self.Beta[-1] = self.new_batch_norm_vars(shape)
        else:
            I = self.multiply(shape)
            Scl = []
            Bta = []
            for i in range(I):
                Scl.append(unpack('f',self.ScaleFile.read(4))[0])
                Bta.append(unpack('f',self.BetaFile.read(4))[0])
            self.Scale[-1], self.Beta[-1] = (
                                                tf.Variable(tf.reshape(tf.constant(Scl, dtype='float32'), shape)), 
                                                tf.Variable(tf.reshape(tf.constant(Bta, dtype='float32'), shape))
                                            )

    def new_weight_variable(self, shape):
        initial = tf.random.truncated_normal(shape, stddev=1e-4)
        return tf.Variable(initial)

    def new_bias_variable(self, shape):
      initial = tf.constant(0.0, shape=shape)
      return tf.Variable(initial)

    def new_batch_norm_vars(self,shape=None):
        scale_init = tf.random.truncated_normal(shape,stddev=1e-4)
        beta_init = tf.random.truncated_normal(shape,stddev=1e-4)
        return tf.Variable(scale_init), tf.Variable(beta_init)

    # Save Weights and Bias
    def save_weights(self):
        try:
            Weights_Bin = open('{}/weights.bin'.format(self.weights_path), 'bw')
            Bias_Bin = open("{}/bias.bin".format(self.weights_path), 'bw')
            for i in range(0, len(self.Weights)):
                Weval = self.Weights[i].eval()
                Wlist = Weval.reshape(Weval.size).tolist()
                Weights_Bin.write(pack('%sf' % len(Wlist), *Wlist))
                Bias_Bin.write(pack('%sf' % len(list(self.Bias[i].eval())), *list(self.Bias[i].eval())))
            Weights_Bin.close()
            Bias_Bin.close()
            Scale_file = open('{}/scale.bin'.format(self.weights_path),'bw')
            Beta_file = open('{}/beta.bin'.format(self.weights_path),'bw')
            for i in range(len(self.Scale)):
                Seval = self.Scale[i].eval()
                Beval = self.Beta[i].eval()
                Slist = Seval.reshape(Seval.size).tolist()
                Blist = Beval.reshape(Beval.size).tolist()
                Scale_file.write(pack('%sf' % len(Slist),*Slist))
                Beta_file.write(pack('%sf' % len(Blist), *Blist))
            Scale_file.close()
            Beta_file.close()
        except FileNotFoundError:
            pass
    # END WEIGHTS AND BIAS FUNCTIONS ##########################################

    # START LAYER FUNCTIONS ###################################################
    def add_layer_with_weight_type(self, Type):
        if Type == 'sigmoid':
            self.Layers[-1] = tf.sigmoid(self.Layers[-1])
        elif Type == 'relu':
            self.Layers[-1] = tf.nn.relu(self.Layers[-1])
        elif Type == 'tanh':
            self.Layers[-1] = tf.tanh(self.Layers[-1])
        elif Type == 'softmax':
            self.Layers[-1] = tf.nn.softmax(self.Layers[-1])
        elif Type != 'linear':
            sys.exit(Type+' is not valid activation fanction')

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    
    def batch_normalization(self,x ,beta, scale, batch_mean, batch_var):
        return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, 1e-3)
    # END LAYER FUNCTIONS #####################################################

    # START USEFUL FUNCTIONS
    def multiply(self, numbers):
        total = 1
        for x in numbers:
            total *= x
        return total

    def LayerShape(self, Layer):
        List = []
        lshape = Layer.get_shape()
        for i in range(1, len(lshape)):
            List.append(int(lshape[i]))
        return List

    # Check if layer syndax ends with ';'
    def CheckEnd(self, Stri):
        if Stri.find(');') != len(Stri)-2:
            sys.exit(Stri+"  Error")

    # PoolSize=1 if new layer is Convolutional , ConvSize=1 if new layer is Pooling
    def HiSize(self, ConvSize=1, PoolSize=1):
        self.Hi = (self.Hi+1-ConvSize)/PoolSize
        if self.Hi == int(self.Hi) and self.Hi >= 2:
            self.Hi = int(self.Hi)
        else:
            sys.exit("Topology error")

    def SetData(self, tr_dt=None, tr_hot=None, ts_dt=None, ts_hot=None, set_data_names=None):
        self.Train_Data = tr_dt
        self.Train_Hot = tr_hot
        self.Test_Data = ts_dt
        self.Test_Hot = ts_hot
        self.Set_Names = set_data_names

    def SetSession(self):
        self.sess = tf.compat.v1.InteractiveSession()

    def Initialize_Vars(self):
        self.sess.run(tf.compat.v1.global_variables_initializer()
)
    # END USEFUL FUNCTIONS

    # START TRAIN #############################################################
    # Te           :Train frequency measurement
    # Tb           :Train batch measurement
    # TrCMatrix    :boolean for Train Confusion Matrix
    # TsCMatrix    :boolean for Test Confusion Matrix
    # test_predict :boolean for Test eval
    def TRAIN(self, Epochs, BatchSize, Te=1, test_predict=True, Tb=40, trainrate=1e-4, TrCMatrix=False, TsCMatrix=False, regularization=False):
        if self.Test_Data is None or self.Test_Hot is None:
            test_predict = False
            TsCMatrix = False
        if Te == 0:
            test_predict = False
        if self.NetworkNewTrainig:
            print(self.DroupoutsProbabilitys)
            CE = tf.reduce_mean(-tf.compat.v1.reduce_sum(self.y_*tf.compat.v1.log(self.Layers[-1]+1e-12), reduction_indices=[1]))
            ### l-2 ####
            reg = 0
            if regularization:
                for W in self.Weights:
                    reg += tf.nn.l2_loss(W)
            CE = CE+0.01*reg
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=trainrate)
            self.train_step = self.opt.minimize(CE)
            self.Initialize_Vars()
            print("#############################################")
            print("#### All tf Variables has been initialized. #")
            print("#############################################")
            self.NetworkNewTrainig = False
        i_epoch = 1
        Batch_Pos = 0  # position index of Train,Hot
        i_batch = 0
        self.DictData = {}
        Return_Mess = "return Lists:\n"
        if Te > 0:
            Return_Mess += "Train Epoch (Loss,Accuracy)\n"
            self.DictData['train'] = {}
            self.DictData['train']['loss'] = []
            self.DictData['train']['accuracy'] = []
        if test_predict:
            Return_Mess += "Test Epoch (Loss,Accuracy)\n"
            self.DictData['test'] = {}
            self.DictData['test']['loss'] = []
            self.DictData['test']['accuracy'] = []
        if Tb > 0:
            Return_Mess += "Batches (Loss,Accuracy)\n"
            self.DictData['batch'] = {}
            self.DictData['batch']['loss'] = []
            self.DictData['batch']['accuracy'] = []
            self.DictData['batch']['info'] = {'batchsize': BatchSize, 'batchfreq': Tb}
        Total_time = 0
        E_time = time.time()  # Epoch time
        while i_epoch <= Epochs:
            # if Batch_Pos+BatchSize > Train_Data,Train_Hot length then copy and merge samples at the beginning of array to the end of array
            D_Batch = np.concatenate((self.Train_Data, self.Train_Data[0:max(Batch_Pos+BatchSize-self.Train_Data.shape[0], 0)]))[Batch_Pos:Batch_Pos+BatchSize]
            L_Batch = np.concatenate((self.Train_Hot, self.Train_Hot[0:max(Batch_Pos+BatchSize-self.Train_Hot.shape[0], 0)]))[Batch_Pos:Batch_Pos+BatchSize]
            self.train_step.run(feed_dict={
                self.Layers[0]: D_Batch,
                self.y_: L_Batch,
                self.keep_prob: self.DroupoutsProbabilitys
            })
            if Tb > 0:
                if i_batch%Tb == 0:
                    lossB = self.LOSS(D_Batch, L_Batch)
                    AccuracyB = self.ACCURACY(D_Batch, L_Batch)
                    Mess = "Epoch "+str(i_epoch)+" -- Batch "+str(i_batch)+": Loss = "+str(lossB)+", Accuracy: "+str(AccuracyB)
                    self.DictData['batch']['loss'].append(lossB)
                    self.DictData['batch']['accuracy'].append(AccuracyB)
                    print(Mess)
            i_batch += 1
            Batch_Pos = (Batch_Pos+BatchSize) % self.Train_Data.shape[0]
            if Batch_Pos < BatchSize:
                E_time = time.time()-E_time
                Total_time += E_time
                mEi = "Epoch "+str(i_epoch)+": "
                mEiSpacing = ''.join([' ']*len(mEi))
                Mess = mEi
                if Te > 0:
                    if i_epoch%Te == 0:
                        tr_i_epoch_Loss = self.LOSS(self.Train_Data, self.Train_Hot)
                        tr_i_epoch_Ac = self.ACCURACY(self.Train_Data, self.Train_Hot)
                        self.DictData['train']['loss'].append(tr_i_epoch_Loss)
                        self.DictData['train']['accuracy'].append(tr_i_epoch_Ac)
                        Mess += "Total Train Loss = "+str(tr_i_epoch_Loss)+", Total Train Accuracy = "+str(tr_i_epoch_Ac)+"\n"+mEiSpacing
                        if test_predict == True:
                            ts_i_epoch_Loss = self.LOSS(self.Test_Data, self.Test_Hot)
                            ts_i_epoch_Ac = self.ACCURACY(self.Test_Data, self.Test_Hot)
                            self.DictData['test']['loss'].append(ts_i_epoch_Loss)
                            self.DictData['test']['accuracy'].append(ts_i_epoch_Ac)
                            Mess = "Total Test Loss = "+str(ts_i_epoch_Loss)+", Total Test Accuracy = "+str(ts_i_epoch_Ac)+"\n"+mEiSpacing+Mess
                Mess += "Epoch Time = "+str(E_time)+" sec. , Total Time = "+str(Total_time)+" sec."
                print(Mess)
                i_epoch += 1
                i_batch = 0
                Batch_Pos = 0
                E_time = time.time()
        if TrCMatrix:
            Return_Mess += "Train Confusion Matrix\n"
            self.DictData['train_confusion_matrix'] = self.ConfusionMatrix(self.Train_Data, self.Train_Hot)
        if TsCMatrix:
            Return_Mess += "Test Confusion Matrix\n"
            self.DictData['test_confusion_matrix'] = self.ConfusionMatrix(self.Test_Data, self.Test_Hot)
        print(Return_Mess)
    # END TRAIN ###############################################################
    # START TRAIN RUSULTS #####################################################

    # Loss and accuracy are calculated per 1000 samples calculated because memory limitation
    def ACCURACY(self, DATA, LABELS):
        correct_prediction = tf.equal(tf.argmax(self.Layers[-1], 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        if DATA.shape[0] < 1000:
            return accuracy.eval(feed_dict={
                    self.Layers[0]: DATA,
                    self.y_: LABELS,
                    self.keep_prob: np.ones((self.DroupoutsProbabilitys.shape[0]))
                })
        Total_Accuracy = 0
        I = int(np.ceil(DATA.shape[0]/1000))
        for i in range(0, I):
            K = min(I-i, 1)
            F = 1000+(DATA.shape[0]-1000*i)*(1-K)
            ac = accuracy.eval(feed_dict={
                self.Layers[0]: DATA[i*F:(i+1)*F-1, :, :, :],
                self.y_: LABELS[i*F:(i+1)*F-1, :],
                self.keep_prob: np.ones((self.DroupoutsProbabilitys.shape[0]))
            })
            Total_Accuracy = (Total_Accuracy*i+ac)/(i+1)  # update total accuracy
        return Total_Accuracy

    def LOSS(self, DATA, LABELS):
        CE = tf.reduce_mean(-tf.compat.v1.reduce_sum(self.y_*tf.compat.v1.log(self.Layers[-1]+1e-12), reduction_indices=[1]))

        if DATA.shape[0] < 1000:
            return CE.eval(feed_dict={
                self.Layers[0]: DATA ,
                self.y_: LABELS,
                self.keep_prob: np.ones((self.DroupoutsProbabilitys.shape[0]))
                })
        Total_Loss = 0
        I = int(np.ceil(DATA.shape[0]/1000))
        for i in range(0, I):
            K = min(I-i, 1)
            F = 1000+(DATA.shape[0]-1000*i)*(1-K)
            loss = CE.eval(feed_dict={
                self.Layers[0]: DATA[i*F:(i+1)*F-1, :, :, :] ,
                self.y_: LABELS[i*F:(i+1)*F-1, :],
                self.keep_prob: np.ones((self.DroupoutsProbabilitys.shape[0]))
            })
            Total_Loss = (Total_Loss*i+loss)/(i+1)  # update total Loss
        return Total_Loss

    def ConfusionMatrix(self, Data, Label):
        Table = np.zeros((len(self.Set_Names), len(self.Set_Names))).astype(int)
        Pop = np.zeros((len(self.Set_Names)))
        for tr in range(Data.shape[0]):
            Score  = self.Layers[-1].eval(feed_dict = {
                     self.Layers[0]: Data[tr].reshape([1]+list(Data.shape[1:])),
                     self.keep_prob : np.ones((self.DroupoutsProbabilitys.shape[0]))
            })
            FindC = np.argmax(Score)
            TrueC = np.argmax(Label[tr])
            Table[TrueC, FindC] += 1
            Pop[TrueC] += 1
        return DF(Table, columns=self.Set_Names, index=self.Set_Names)

    def PrintTrainLossAccuracy(self):
        print("Total train Accuracy = %f"%(self.ACCURACY(self.Train_Data, self.Train_Hot)))
        print("Total train Loss = %f"%(self.LOSS(self.Train_Data, self.Train_Hot)))

    def PrintTestLossAccuracy(self):
        print("Total test Accuracy = %f"%(self.ACCURACY(self.Test_Data, self.Test_Hot)))
        print("Total test Loss = %f"%(self.LOSS(self.Test_Data, self.Test_Hot)))
    # END TRAIN RUSULTS #######################################################


    # START CSV ###############################################################
    # DictDir :name directory for results
    # id      :subdirectory where the train results are saved
    def SaveDictData(self, id, DictDir=None):
        if DictDir == None:
            DictDir = self.prdir+'_RS'
        elif not(os.path.isdir('../CSVRESULTS/'+DictDir)):
            os.mkdir('../CSVRESULTS/'+DictDir)
        if not(os.path.isdir('../CSVRESULTS/'+DictDir+'/'+id)):
            os.mkdir('../CSVRESULTS/'+DictDir+'/'+id)

        path = '../CSVRESULTS/'+DictDir+'/'+id+'/'
        if 'train' in self.DictData:
            C = csv.writer(open(path+"outputTR.csv", 'w'))
            C.writerow(self.DictData['train']['loss'])
            C.writerow(self.DictData['train']['accuracy'])
        if 'test' in self.DictData:
            C = csv.writer(open(path+"outputTS.csv", 'w'))
            C.writerow(self.DictData['test']['loss'])
            C.writerow(self.DictData['test']['accuracy'])
        if 'batch' in self.DictData:
            C = csv.writer(open(path+"outputB.csv", 'w'))
            C.writerow(self.DictData['batch']['loss'])
            C.writerow(self.DictData['batch']['accuracy'])
            C.writerow([self.DictData['batch']['info']['batchsize'], self.DictData['batch']['info']['batchfreq']])
        if 'train_confusion_matrix' in self.DictData:
            self.DictData['train_confusion_matrix'].to_csv(path+'TrainConfusionMatrix.csv', sep=',')
            if 'test' in self.DictData:
                self.DictData['test_confusion_matrix'].to_csv(path+'TestConfusionMatrix.csv', sep=',')

    def LoadCSVtoDict(self, id, DictDir=None):
        def OpenCSV(filename):
            C = csv.reader(open(filename+".csv", 'r'))
            L = []
            for r in C:
                L.append([float(i) for i in r])
            return L
        if DictDir is None:
            DictDir = self.prdir+'_RS'
        path = '../CSVRESULTS/'+DictDir+'/'+id+'/'
        self.DictData = {}
        if os.system('ls '+path+'outputTR.csv') != 512:  # True
            self.DictData['train'] = {}
            self.DictData['train']['loss'], self.DictData['train']['accuracy'] = OpenCSV(path+'outputTR')
        if os.system('ls '+path+'outputTS.csv') != 512:
            self.DictData['test'] = {}
            self.DictData['test']['loss'], self.DictData['test']['accuracy'] = OpenCSV(path+'outputTS')
        if os.system('ls '+path+'outputB.csv') != 512:
            self.DictData['batch'] = {}
            self.DictData['batch']['loss'], self.DictData['batch']['accuracy'], self.DictData['batch']['info'] = OpenCSV(path+'outputB')
            self.DictData['batch']['info'] = {'batchsize': self.DictData['batch']['info'][0], 'batchfreq': self.DictData['batch']['info'][1]}
        if os.system('ls '+path+'TrainConfusionMatrix.csv') != 512:
            self.DictData['train_confusion_matrix'] = DF.from_csv(path+'TrainConfusionMatrix.csv')
        if os.system('ls '+path+'TestConfusionMatrix.csv') != 512:
            self.DictData['test_confusion_matrix'] = DF.from_csv(path+'TestConfusionMatrix.csv')

    # END CSV #################################################################
    # START VISUALIZATION #####################################################
    # prints two plots one with Train-Test Loss and other with Train-Test Accuracy
    def TrainTestPlot(self):
        if 'train' in self.DictData and 'test' in self.DictData:
            plt.subplot(1, 2, 1)
            plt.plot(self.DictData['train']['loss'], label="Train Loss")
            plt.plot(self.DictData['test']['loss'], label="Test Loss")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(self.DictData['train']['accuracy'], label="Train Accuracy")
            plt.plot(self.DictData['test']['accuracy'], label="Test Accuracy")
            plt.legend()
            plt.show()

    # prints maximum three plots with Train Loss-Accuracy, Test Loss-Accuracy, Batch Loss-Accuracy
    def DictDataPlot(self, PlotList:list=['train', 'test', 'batch']):
        fig = plt.figure()
        Ax = []
        i = 0  # plot counter
        for lf in PlotList:
            if lf == 'train' and lf in self.DictData:
                Ax.append(fig.add_axes([0.05, 0.1, 0.4, 0.7]))
                Ax[-1].set_title('Train. Epochs '+str(len(self.DictData['train']['loss'])))
                L = Ax[-1].plot(self.DictData['train']['loss'], label='Loss')
                AC = Ax[-1].plot(self.DictData['train']['accuracy'], label='Accuracy')
                Ax[-1].legend()
                i += 1
            elif lf == 'test' and lf in self.DictData:
                Ax.append(fig.add_axes([0.5, 0.1, 0.4, 0.7]))
                Ax[-1].set_title('Test. Epochs '+str(len(self.DictData['test']['loss'])))
                L = Ax[-1].plot(self.DictData['test']['loss'], label='Loss')
                AC = Ax[-1].plot(self.DictData['test']['accuracy'], label='Accuracy')
                Ax[-1].legend()
                i += 1
            elif lf == 'batch' and lf in self.DictData:
                Ax.append(fig.add_axes([0.9, 0.1, 0.4, 0.7]))
                Ax[-1].set_title('Batch. Batch Size '+str(self.DictData['batch']['info']['batchsize'])+', Batch Friquency Mesurement '+str(self.DictData['batch']['info']['batchfreq']))
                L = Ax[-1].plot(self.DictData['batch']['loss'], label='Loss')
                AC = Ax[-1].plot(self.DictData['batch']['accuracy'], label='Accuracy')
                Ax[-1].legend()
                i += 1
        # here are calculated the plots possessions and sizes
        if i == 1:
            Ax[0].set_position([0.05, 0.05, 0.90, 0.90])
            pass
        elif i == 2:
            Ax[0].set_position([0.05, 0.15, 0.45, 0.65])
            Ax[1].set_position([0.55, 0.15, 0.45, 0.65])
        elif i == 3:
            Ax[0].set_position([0.05, 0.55, 0.45, 0.40])
            Ax[1].set_position([0.55, 0.55, 0.45, 0.40])
            Ax[2].set_position([0.275, 0.05, 0.45, 0.40])
        plt.show()
# END VISUALIZATION ###########################################################
# END CLASS####################################################################
