# NN-Tool
## Simple Artificial Neural Network tool written in python3 - Tensorflow

* #### Package Requirements
  1. numpy
  2. matplotlib
  3. pandas
  4. tensorflow
  5. struct
  6. urllib
  7. scipy

* #### What you can do?
  1. Write the Neural Network Topology in .txt file.
  2. View Training & Testing results.
  3. Save Views and Training Progress.
* #### Why NN-Tool?
  1. Simple architectures - *for now*.
  2. Little piece of code - *The reason for the project*.
* #### Set Up
  1. Clone git
  ```bush
  $ git clone https://github.com/EfMichalis/NN-Tool
  ```
  2. Go to NN-Tool folder
  ```bush
  $ cd path/to/NN-Tool
  ```
  3. Run  *setup.py*
  ```bush
  NN-Tool$ python3 setup.py
  ```
* #### NN-Tool Structure
  ```
    NN-Tool
    ├── CSVRESULTS
    ├── DATASETS
    ├── LIBRARY
    │   ├── DATASETS.py
    │   └── NEURAL_NETWORK_TOOL.py
    ├── SCRIPTS
    ├── TOPOLOGYS
    ├── newproject.py
    └── setup.py

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
Create new Project and give **IRIS** as project name:
``` bush
NN-Tool$ python3 newproject.py
Give Project Name: IRIS
```
```bush
NN-Tool
├── CSVRESULTS
|   └──IRIS_RS
├── DATASETS
├── LIBRARY
│   ├── DATASETS.py
│   └── NEURAL_NETWORK_TOOL.py
├── SCRIPTS
│   └── IRIS_SC.py
├── TOPOLOGYS
|   └── IRIS_TP
|      └── topology.txt
├── newproject.py
└── setup.py
```
Copy this to *topology.txt*:
```txt
Input(4,1,1);
Fc(1024,tanh);
Fc(2056,tanh);
Fc(512,tanh);
Fc(3,softmax);
```

Go to SCRIPTS folder
```bush
NN-Tool$ cd SCRIPTS
```
Open *IRIS_SC.py*,with first line :
``` python
import sys;sys.path.append('../LIBRARY');from NEURAL_NETWORK_TOOL import *
 ```
Read Iris Dataset:
``` python
# 150 Examples, by default 120 Train Examples & 30 Test Examples
# The IRISdata() has Number of Train Examples as input argument
Train_Examples,Train_Labels,Test_Examples,Test_Labels,Set_Names=IRISdata()
```
Prepare the model before the training:
```python
Net = NNtool() # Topology Directory  : IRIS_TP
Net.SetData(Train_Examples,Train_Labels,Test_Examples,Test_Labels,Set_Names)
Net.SetSession() #Ready up
```
Train Model:
```python
BatchSize=20
N_Epoch=25

#TRAIN inputs, dont make sence. Read LIBRARY/NEURAL_NETWORK_TOOL.py line 277
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
experiment_folder='1' # This is first traning experiment so we create folder 1
Net.SaveDictData(experiment_folder) # Results Directory : IRIS_RS
```
If everything is fine we can read the results by the files:
```python
Net.LoadCSVtoDict(experiment_folder)
Net.TrainTestPlot()
Net.DictDataPlot()
print(Net.DictData['train_predict_table'])
```
Adding all pieces of code:
```python
import sys;sys.path.append('../LIBRARY');from NEURAL_NETWORK_TOOL import *

# 150 Examples, by default 120 Train Examples & 30 Test Examples
# The IRISdata() has Number of Train Examples as input argument
Train_Examples,Train_Labels,Test_Examples,Test_Labels,Set_Names=IRISdata()


Net = NNtool() # Topology Directory  : IRIS_TP
Net.SetData(Train_Examples,Train_Labels,Test_Examples,Test_Labels,Set_Names)
Net.SetSession() #Ready up

BatchSize=20
N_Epoch=25

#TRAIN inputs, dont make sence. Read LIBRARY/NEURAL_NETWORK_TOOL.py line 277
Net.TRAIN(N_Epoch,BatchSize,Tb=3,Te=1,test_predict=True)
Net.SaveWeights()

Net.PrintTrainLossAccuracy()
Net.PrintTestLossAccuracy()
Net.TrainTestPlot()
Net.DictDataPlot()
print(Net.DictData['train_predict_table'])

experiment_folder='1' # This is first traning experiment so we create folder 1
Net.SaveDictData(experiment_folder) # Results Directory : IRIS_RS

Net.LoadCSVtoDict(experiment_folder)
Net.TrainTestPlot()
Net.DictDataPlot()
print(Net.DictData['train_predict_table'])
```
The NN-Tool tree after this example should be like this:
```
NN-Tool
├── CSVRESULTS
│   └── IRIS_RS
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
│   └── IRIS_SC.py
└── TOPOLOGYS
    └── IRIS_TP
        ├── BiasBin.Bin
        ├── topology.txt
        └── WightsBin.Bin
```
