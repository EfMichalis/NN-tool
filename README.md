# NN-Tool
## Simple Artificial Neural Network tool written in python3 - Tensorflow

* #### Package Requirements
  1. numpy
  2. matplotlib
  3. pandas
  4. tensorflow

* #### What you can do?
  1. Write the Neural Network Topology in .txt file.
  2. View Training & Testing results.
  3. Save Views and Training Progress.
* #### Why NN-tool?
  1. Simple architectures - *for now*.
  2. Little piece of code - *The reason for the project*.
* #### Set Up
  1. Clone git
  ```bush
  $ git clone https://github.com/EfMichalis/NN-tool
  ```
  2. Go to NN-tool folder
  ```bush
  $ cd path/to/NN-tool
  ```
  3. Create *SCRIPTS* folder , *TOPOLOGYS* folder and *CSVRESULTS* folder
  ```bush
  NN-Tool$ mkdir SCRIPTS TOPOLOGYS CSVRESULTS
  ```
* #### NN-tool Structure
  ```
    NN-TOOL
    ├── CSVRESULTS
    ├── DATASETS
    │   └── IRIS
    │       └── IRIS_DATA.bin
    ├── LIBRARY
    │   ├── DATASETS.py
    │   └── NEURAL_NETWORK_TOOL.py
    ├── SCRIPTS
    └── TOPOLOGYS
  ```
* #### Topology Syndax
Layer base syndax:<br>
  <b>LayerType(param0,param1,...);</b><br>
Layer Types:
<ol>
<li>INPUT : <b>Input(X_size,Y_size,Z_size)</b></li>
<li>CONVOLUTIONAL : <b>Conv(block_size,Z_size,Act. Function)</b></li>
<li>POOLING : <b>Pool(block_size,PoolMethod)</b></li>
<li>FULL CONNECTED : <b>Fc(X_size,Act. Function)</b></li>
<li>DROPOUT : <b>Dropout(probability%)</b></li>
</ol>
Syndax rules:
<ol>
  <li> <i>INPUT musth be the first layer</i> </li>
  <li> <i>X_size,Y_size,Z_size >= 1</i> </li>
  <li> <i>CONVOLUTIONAL Act. Function:</i>
    <ul>
      <li> <i>sigmoid</i> </li>
      <li> <i>tanh</i> </li>
      <li> <i>relu</i> </li>
      <li> <i>linear</i> </li>
    </ul>
  </li>
  <li> <i>POOL PoolMethod:</i>
    <ul>
      <li> <i>max</i> </li>
      <li> <i>min</i> </li>
      <li> <i>avg</i> </li>
    </ul>
  </li>
  <li> <i>FULL CONNECTED Act. Function:</i>
    <ul>
      <li> <i>All CONVOLUTIONAL Act. Functions</i> </li>
      <li> <i>softmax</i> </li>
    </ul>
  </li>
  <li> <i>DROPOUT :</i>
    <ul>
      <li> <i>probability between  [0,100]</i> </li>
    </ul>
  </li>
  <li> <i>Layers Relations:</i>
      <li> <i>CONVOLUTIONAL and POOLING must be precedes </i>
        <ol>
          <li> <i>INPUT-(X_size,Y_size>=2,X_size = Y_size)</i> </li>
          <li> <i>Other CONVOLUTIONAL</i> </li>
          <li> <i>POOLING</i> </li>
          <li> <i>DROPOUT ,where the last non-dropout is one of the above</i> </li>
        </ol>
      </li>
      <li> <i>This expression must be natural number: <br>(This_Layer_X_size+1-Conv_X_size)/Pool_X_size, where, if next layer is CONVOLUTIONAL then Pool_X_size=1 or if next layer is POOLING then Conv_X_size=1<br>To culculate the sizes by the hand for valid topology start from the INPUT's sizes</i> <b>(This expression is calculated only for CONVOLUTIONAL and POOLING Layers.If the next layer is FULL CONNECTED then NN-Tool is flattening the current layer)</b>
      </li>
</li>


### EXAMPLE
We use [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), every example is a 1D vector size 4 and the number of classes is 3.<br><br>
The first thing we do ,is to make a topology folder *iris_topNN* :
``` bush
NN-TOOL$ mkdir TOPOLOGYS/iris_topNN
```
Now create *topology.txt*:
``` bush
NN-TOOL$ touch TOPOLOGYS/iris_topNN/topology.txt
```
Copy this to *topology.txt*:
```txt
Input(4,1,1);
Fc(1024,tanh);
Fc(2056,tanh);
Fc(512,tanh);
Fc(3,softmax);
```
create result folder in CSVRESULTS *iris*
``` bush
NN-TOOL$ mkdir CSVRESULTS/iris
```
This is first traing experiment so we create folder *1*
``` bush
NN-TOOL$ mkdir CSVRESULTS/iris/1
```
In SCRIPTS folder, we will create a script for model training:

``` bush
NN-TOOL$ touch SCRIPTS/iris_exmaple.py
```
Go to SCRIPTS folder
```bush
NN-TOOL$ cd SCRIPTS
```
Open *iris_exmaple.py* and call NN-Tool module:
``` python
import sys
sys.path.append('../LIBRARY')
from NEURAL_NETWORK_TOOL import *
 ```
Read Iris Dataset:
``` python
# 150 Examples, by default 120 Train Examples & 30 Test Examples
# The IRISdata() has Number of Train Examples as input argument
Train_Examples,Train_Labels,Test_Examples,Test_Labels,Set_Names=IRISdata()
```
Prepare the model before the training:
```python
Dir="iris_topNN" #Topology Directory
Net = NNtool()
Net.buildnet(Dir)
Net.SetData(Train_Examples,Train_Labels,Test_Examples,Test_Labels,Set_Names)
Net.SetSession() #Ready up
```
Train Model:
```python
BatchSize=20
N_Epoch=40

#Inputs, dont make sence. Read LIBRARY/NEURAL_NETWORK_TOOL.py line 277
Net.TRAIN(N_Epoch,BatchSize,Tb=3,Te=1,test_predict=True)
```
Save our progress:
```python
Net.SaveWeights()
```
Show Results:
```python
Net.PrintTrainLossAccuracy()
Net.PrintTestLossAccuracy()
Net.TrainTestPlot()
Net.DictDataPlot()
print(Net.DictData['train_predict_table'])
```
Save Results:
```python
Iris_Resultfolder='iris'
experiment_folder='1'
Net.SaveDictData(Iris_Resultfolder,experiment_folder)
```
If everything is fine we can read the results by the files:
```python
Net.LoadCSVtoDict(Iris_Resultfolder,experiment_folder)
Net.TrainTestPlot()
Net.DictDataPlot()
print(Net.DictData['train_predict_table'])
```
Adding all pieces of code:
```python
# iris_exaple.py
import sys
sys.path.append('../LIBRARY')
from NEURAL_NETWORK_TOOL import *

# 150 Examples, by default 120 Train Examples & 30 Test Examples
# The IRISdata() has Number of Train Examples as input argument
Train_Examples,Train_Labels,Test_Examples,Test_Labels,Set_Names=IRISdata()

Dir="iris_topNN" #Topology Directory
Net = NNtool()
Net.buildnet(Dir)
Net.SetData(Train_Examples,Train_Labels,Test_Examples,Test_Labels,Set_Names)
Net.SetSession() #Ready up

BatchSize=20
N_Epoch=40

#Inputs, dont make sence. Read LIBRARY/NEURAL_NETWORK_TOOL.py line 277
Net.TRAIN(N_Epoch,BatchSize,Tb=3,Te=1,test_predict=True)
Net.SaveWeights()

Net.PrintTrainLossAccuracy()
Net.PrintTestLossAccuracy()
Net.TrainTestPlot()
Net.DictDataPlot()
print(Net.DictData['train_predict_table'])

Iris_Resultfolder='iris'
experiment_folder='1'
Net.SaveDictData(Iris_Resultfolder,experiment_folder)

Net.LoadCSVtoDict(Iris_Resultfolder,experiment_folder)
Net.TrainTestPlot()
Net.DictDataPlot()
print(Net.DictData['train_predict_table'])
```
the NN-tool tree after this example should be like this:
```
NN-TOOL
├── CSVRESULTS
│   └── iris
│       └── 1
│           ├── outputB.csv
│           ├── outputTR.csv
│           ├── outputTS.csv
│           ├── TestPredictionTable.csv
│           └── TrainPredictionTable.csv
├── DATASETS
│   └── IRIS
│       └── IRIS_DATA.bin
├── LIBRARY
│   ├── DATASETS.py
│   └── NEURAL_NETWORK_TOOL.py
├── SCRIPTS
│   └── iris_example.py
└── TOPOLOGYS
    └── iris_topNN
        ├── BiasBin.Bin
        ├── topology.txt
        └── WightsBin.Bin
```
