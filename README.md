[![CircleCI](https://circleci.com/gh/EfMichalis/NN-Tool.svg?style=svg)](https://circleci.com/gh/EfMichalis/NN-Tool)

# nntool
## Artificial Neural Network(ANN) Freamwork in python3 to read ANN topology from txt file

* #### Package Requirements
  1. docopt
  2. tensorflow
  3. scipy
  4. matplotlib
  5. pandas
  6. scikit-image
  7. numpy

#### Topology Syndax

Base Layer syndax:
LayerType(param0,param1,...);
Layer Types:
1. INPUT : Input(X_size,Y_size,Z_size)
2. CONVOLUTIONAL : Conv(block_size,Z_size,Act. Function)
3. POOLING : Pool(block_size,PoolMethod)
4. FULL CONNECTED : Fc(X_size,Act. Function)
5. DROPOUT : Dropout(probability%)
6. BATCH NORMALIZATION : BatchNorm()

Syndax:
1. INPUT must be the first layer
* X_size,Y_size,Z_size >= 1
2. CONVOLUTIONAL Act. Function:
* sigmoid
* tanh
* relu
* linear

4. POOL PoolMethods:
* max
* min
* avg

5. FULL CONNECTED Act. Function:
* All CONVOLUTIONAL Act. Functions
* softmax

DROPOUT
* probability between [0,100]

IRiS Dataset Example:

Input(4,1,1);<br>
Fc(1024,tanh);<br>
Fc(2056,tanh);<br>
Fc(512,tanh);<br>
Fc(3,softmax);
