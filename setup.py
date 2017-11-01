import os
if(not(os.path.exists('CSVRESULTS'))):
    os.makedirs('CSVRESULTS')

if(not(os.path.exists('DATASETS'))):
    os.makedirs('DATASETS')

if(not(os.path.exists('SCRIPTS'))):
    os.makedirs('SCRIPTS')

if(not(os.path.exists('TOPOLOGYS'))):
    os.makedirs('TOPOLOGYS')

if(not(os.path.exists('newproject.py'))):
    with open('newproject.py', 'w') as Pyfile:
        Pyfile.write("""import os;PrName=input('Give Project Name: ');SCfile = open('SCRIPTS/'+PrName+'_SC.py','w');SCfile.write("import sys;sys.path.append('../LIBRARY');from NEURAL_NETWORK_TOOL import *");SCfile.close();os.makedirs('TOPOLOGYS/'+PrName+'_TP');os.system('touch TOPOLOGYS/'+PrName+'_TP/topology.txt');os.makedirs('CSVRESULTS/'+PrName+'_RS')""")
