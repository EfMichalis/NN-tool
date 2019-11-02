# DATASETS.py
import numpy as np
from struct import *
import os
import urllib.request
import matplotlib.image as mpimg
from scipy.misc import imresize
import random


# MNIST ###### 60000 Trains sample 10000 Test samples
def mnist_dawnload(get=True):
    if(not(os.path.exists('.DATASETS/MNIST'))):
        print("Download Train Images")
        urllib.request.urlretrieve(
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "train-images-idx3-ubyte.gz"
        )
        print("Dawnload Train Labels")
        urllib.request.urlretrieve(
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "train-labels-idx1-ubyte.gz"
        )
        print("Dawnload Test Images")
        urllib.request.urlretrieve(
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "t10k-images-idx3-ubyte.gz"
        )
        print("Dawnload Test Labels")
        urllib.request.urlretrieve(
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        )
        print("Dawnload complete")
        print("Create NN-tool/lDATASETS/MNIST folder")
        os.system('mkdir .DATASETS/MNIST')

        print("decompress MNIST Dataset")
        Decom = "mv *.gz .DATASETS/MNIST ; "
        Decom += "gzip -d .DATASETS/MNIST/*.gz"
        os.system(Decom)

        Rename = "mv .DATASETS/MNIST/train-images-idx3-ubyte" \
                 " .DATASETS/MNIST/TrainImages.bin ;"
        Rename += " mv .DATASETS/MNIST/train-labels-idx1-ubyte" \
                  " .DATASETS/MNIST/TrainLabels.bin ;"
        Rename += " mv .DATASETS/MNIST/t10k-images-idx3-ubyte" \
                  " .DATASETS/MNIST/TestImages.bin ;"
        Rename += " mv .DATASETS/MNIST/t10k-labels-idx1-ubyte" \
                  " .DATASETS/MNIST/TestLabels.bin"
        os.system(Rename)
    else:
        print('MNIST data set exist')
    if get:
        return mnist()


def mnist(trn=60000, tsn=10000):
    def Get_From_Bin_File(Numb, LabelBin, ImageBin):
        print(
            "Read Labels from %s.\nRead Images from %s" %
            (LabelBin, ImageBin))
        ImBinReader = open(ImageBin, 'br')
        LabelBinRader = open(LabelBin, 'br')
        ImBinReader.read(16)
        LabelBinRader.read(8)
        Image = np.zeros((Numb, 28, 28, 1))
        Label = np.zeros((Numb, 10))
        a = np.arange(10)
        for i in range(0, Numb):
            Image[i, :, :, :] = np.array(
                [list(unpack_from("<%iB" % (784), ImBinReader.read(784)))]).reshape(28, 28, 1) / 255
            Label[i, :] = np.sign(-np.fabs(a - LabelBinRader.read(1)[0])) + 1
        ImBinReader.close()
        LabelBinRader.close()
        return Image, Label
    if(not(os.path.exists('.DATASETS/MNIST'))):
        print('MNIST data set dont exist\n Download MNIST data-set')
        return mnist_dawnload(get=True)
    else:
        TrainImagePath = ".DATASETS/MNIST/TrainImages.bin"
        TrainLabelPath = ".DATASETS/MNIST/TrainLabels.bin"
        # 1 <= trainNumb <= 60000
        Numb = trn
        TrainIm, TrainLabel = Get_From_Bin_File(
            Numb, TrainLabelPath, TrainImagePath)

        TestImagePath = ".DATASETS/MNIST/TestImages.bin"
        TestLabelPath = ".DATASETS/MNIST/TestLabels.bin"
        # 1 <= testNumb <= 10000
        Numb = tsn
        TestIm, TestLabel = Get_From_Bin_File(
            Numb, TestLabelPath, TestImagePath)
        Set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        return TrainIm, TrainLabel, TestIm, TestLabel, Set


# CIFAR-10
def cifar10_dawnload(get=True):
    if(not(os.path.exists('.DATASETS/CIFAR-10'))):
        print("Dawnload CIFAR-10")
        urllib.request.urlretrieve(
            "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
            "cifar-10-binary.tar.gz"
        )
        print("Dawnload complete")
        print("Create NN-C/DATASETS/MNIST folder")
        print("decompress CIFAR-10 Dataset")
        Decom = " tar -xzvf cifar-10-binary.tar.gz ;"
        Decom += " rm cifar-10-binary.tar.gz "
        os.system(Decom)

        print("Move CIFAR-10 to NN-tool/DATASETS")
        os.system("mv cifar-10-batches-bin CIFAR-10; mv CIFAR-10 .DATASETS/")
    else:
        print('CIFAR-10 dataset exists')
    if get:
        return cifar10()


def cifar10(trn=50000, tsn=10000):
    if(not(os.path.exists('.DATASETS/CIFAR-10'))):
        print('CIFAR-10 dataset dont exist')
        return cifar10_dawnload(get=True)
    else:
        def cifar10set_data_and_labels(Numb, BinFile):
            FileReader = open(BinFile, 'rb')
            a = np.arange(10)
            Image = np.zeros((Numb, 32, 32, 3))
            Label = np.zeros((Numb, 10))
            i = 0
            for i in range(0, Numb):
                Label[i, :] = np.sign(-np.fabs(a - FileReader.read(1)[0])) + 1
                Image[i, :, :, 0] = np.array(
                    [list(unpack_from("<%iB" % (1024), FileReader.read(1024)))]
                ).reshape(32, 32)

                Image[i, :, :, 1] = np.array(
                    [list(unpack_from("<%iB" % (1024), FileReader.read(1024)))]
                ).reshape(32, 32)

                Image[i, :, :, 2] = np.array(
                    [list(unpack_from("<%iB" % (1024), FileReader.read(1024)))]
                ).reshape(32, 32)

                Image[i, :, :, :] = Image[i, :, :, :] / 255
            FileReader.close()
            return Image, Label
        # Train
        Set = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"]
        # 1 <= trainNumb <= 50000
        Numb = trn
        BinFile = ".DATASETS/CIFAR-10/data_batch_0.bin"
        I = int(np.ceil(Numb / 10000))
        TrainIm = np.zeros((1, 32, 32, 3))
        TrainLabel = np.zeros((1, 10))
        for i in range(1, I + 1):
            print('Read Train batch file ' + str(i))
            BinFile = BinFile.replace("_" + str(i - 1), "_" + str(i))
            K = min(I - i, 1)
            F = 10000 + (Numb - 10000 * i) * (1 - K)
            trIm, trLb = cifar10set_data_and_labels(F, BinFile)
            TrainIm = np.concatenate((TrainIm, trIm))
            TrainLabel = np.concatenate((TrainLabel, trLb))
        TrainIm = TrainIm[1:TrainIm.shape[0], :, :, :]
        TrainLabel = TrainLabel[1:TrainLabel.shape[0], :]
        # 1 <= TestNumb <= 10000
        Numb = tsn
        BinFile = ".DATASETS/CIFAR-10/test_batch.bin"
        print('Read Test file')
        TestIm, TestLabel = cifar10set_data_and_labels(Numb, BinFile)
        return TrainIm, TrainLabel, TestIm, TestLabel, Set


def chars74k_num_caps_fonts_dawnload(get=True):
    if(not(os.path.exists('.DATASETS/CHARS74K_NUM_CAPS_FONTS'))):
        print("Download CHARS74K fonts")
        urllib.request.urlretrieve(
            "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz",
            "EnglishFnt.tgz"
        )
        print("Dawnload complete")
        print('create .bin file')
        os.system("tar -xvzf EnglishFnt.tgz ; rm EnglishFnt.tgz")
        # Set Names(
        #            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        #            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        #            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        #            'U', 'V', 'W', 'X', 'Y', 'Z'
        #          )
        DataList = []
        RootDir = 'English/Fnt'
        for dir in os.listdir(RootDir):
            if int(dir[-3:]) >= 1 and int(dir[-3:]) <= 36:
                for file in os.listdir(RootDir + "/" + dir):
                    DataList.append([dir, RootDir + "/" + dir + "/" + file])

        random.shuffle(DataList)
        N = len(DataList)  # 36576 for now
        with open('CHARS74K_NUM_CAPS_FONTS.bin', 'bw') as BinWrite:
            for D in DataList:
                Class = int(D[0][-3:]) - 1
                img = mpimg.imread(D[1])
                img = imresize(img, (28, 28), 'bicubic')
                img = img.reshape(784).tolist()
                BinWrite.write(pack('%sB' % 1, *[Class]))
                BinWrite.write(pack('%sB' % 784, *img))
        print(
            'move CHARS74K_NUM_CAPS_FONTS.bin to' /
            ' NN-tool/DATASETS/CHARS74K_NUM_CAPS_FONTS'
        )
        os.system(
            'rm -rf English ; mkdir .DATASETS/CHARS74K_NUM_CAPS_FONTS ;' /
            ' mv CHARS74K_NUM_CAPS_FONTS.bin .DATASETS/CHARS74K_NUM_CAPS_FONTS'
        )
    else:
        print('CHARS74K_NUM_CAPS_FONTS dataset exists')

    if get:
        return chars74k_num_caps_fonts()


def chars74k_num_caps_fonts(trn=36576):
    if(not(os.path.exists('.DATASETS/CHARS74K_NUM_CAPS_FONTS'))):
        print('CHARS74K_NUM_CAPS_FONTS dataset doesnt exist')
        return chars74k_num_caps_fonts_dawnload()
    else:
        Path = '.DATASETS/CHARS74K_NUM_CAPS_FONTS/CHARS74K_NUM_CAPS_FONTS.bin'
        RDataSet = open(Path, 'br')
        TrainIm = np.zeros((trn, 28, 28, 1))
        TrainLabel = np.zeros((trn, 36))
        for i in range(0, trn):
            Laybel = RDataSet.read(1)[0]
            TrainLabel[i, Laybel] = 1
            TrainIm[i] = np.array(
                [list(unpack_from("<%iB" % (784), RDataSet.read(784)))]
            ).reshape(28, 28, 1) / 255

        TestIm = np.zeros((36576 - trn, 28, 28, 1))
        TestLabel = np.zeros((36576 - trn, 36))
        for i in range(36576 - trn):
            Laybel = RDataSet.read(1)[0]
            TestLabel[i, Laybel] = 1
            TestIm[i] = np.array(
                [list(unpack_from("<%iB" % (784), RDataSet.read(784)))]
            ).reshape(28, 28, 1) / 255

        RDataSet.close()
        Set = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z'
              ]

        return TrainIm, TrainLabel, TestIm, TestLabel, Set


# IRIS DATASET 150 Examples
def iris_dawnload(get=True):
    if(not(os.path.exists('.DATASETS/IRIS'))):
        print('Download IRIS dataset')
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
            "iris.data"
        )
        print('Download complete')
        AR = np.zeros((150, 5))
        with open('iris.data', 'r') as f:
            i = 0
            for Line in f:
                line = Line[:-1].split(',')
                line[0] = float(line[0])
                line[1] = float(line[1])
                line[2] = float(line[2])
                line[3] = float(line[3])
                if(line[4] == 'Iris-setosa'):
                    line[4] = 0.0
                elif(line[4] == 'Iris-versicolor'):
                    line[4] = 1.0
                elif(line[4] == 'Iris-virginica'):
                    line[4] = 2.0

                AR[i] = np.array(line)
                i += 1
                if i == 150:
                    break
        AR = np.random.permutation(AR)
        with open('IRIS_DATA.bin', 'bw') as IRIS_DATA:
            for i in range(0, 150):
                ar2li = AR[i].tolist()
                IRIS_DATA.write(pack('%4sf' % len(ar2li), *ar2li))
        os.system(
            'mkdir .DATASETS/IRIS;' \
            ' mv IRIS_DATA.bin .DATASETS/IRIS/;' \
            ' rm iris.data'
        )
    else:
        print('IRIS dataset exists')
    if get:
        return iris()


def iris(trn=120):
    if(not(os.path.exists('.DATASETS/IRIS'))):
        print('IRIS dataset doesnt exist')
        return iris_dawnload(get=True)
    else:
        # 0<=TR_data<=150
        N = trn
        # Testing samples 150-N
        a = np.arange(3)
        IRIS_DATA = open('.DATASETS/IRIS/IRIS_DATA.bin', 'br')

        TrainIm = np.zeros((N, 4))
        TrainLabel = np.zeros((N, 3))
        for i in range(0, N):
            TrainIm[i] = np.array(
                [list(unpack_from("<4f", IRIS_DATA.read(16)))])
            TrainLabel[i] = np.sign(-np.fabs(a-unpack_from("f", IRIS_DATA.read(4))[0])) + 1

        TestIm = np.zeros((150 - N, 4))
        TestLabel = np.zeros((150 - N, 3))
        for i in range(0, 150 - N):
            TestIm[i] = np.array(
                [list(unpack_from("<4f", IRIS_DATA.read(16)))])
            TestLabel[i] = np.sign(-np.fabs(a-unpack_from("f", IRIS_DATA.read(4))[0])) + 1
        IRIS_DATA.close()
        Set = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        return TrainIm.reshape(N, 4, 1, 1), TrainLabel, TestIm.reshape(150-N, 4, 1, 1), TestLabel, Set
