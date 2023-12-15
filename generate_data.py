import numpy as np
from diagram import diagram
import argparse
import os

def makeTask1Sets(N, dest):
    matrices = []
    labels = []
    for _ in range(int(N)):
        dgrm = diagram.createDiagram()
        danger = np.array(dgrm.layWires())
        matrices.append(dgrm.data.flatten())
        labels.append(danger)
    matrices = np.array(matrices)
    labels = np.array(labels)

    # In case the necessary directory hasn't been made yet
    os.makedirs(args.dest, exist_ok=True)
    np.save(arr=matrices,file=f'{dest}/data.npy',allow_pickle=True)
    np.save(arr=labels,file=f'{dest}/labels.npy',allow_pickle=True)

def makeTask2Sets(N, dest):
    matrices = []
    labels = []

    count=0
    while(count<int(N)):
        dgrm = diagram.createDiagram()
        danger = np.array(dgrm.layWires())
        print(f"my danger: {danger}")
        if danger[1] != 0:
            matrices.append(dgrm.data.flatten())
            labels.append(danger)
            count+=1

    matrices = np.array(matrices)
    labels = np.array(labels)

    # In case the necessary directory hasn't been made yet
    os.makedirs(args.dest, exist_ok=True)
    np.save(arr=matrices,file=f'{dest}/data.npy',allow_pickle=True)
    np.save(arr=labels,file=f'{dest}/labels.npy',allow_pickle=True)

#Usage: python3 generate_data.py --N --dest
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='In this script we will essentially create all the training data and store their labels')
    parser.add_argument('--N',help= 'Number of diagrams we want to generate')
    parser.add_argument('--dest', help='folder where we will store all the data')
    args = parser.parse_args()

    makeTask1Sets(args.N, args.dest)

