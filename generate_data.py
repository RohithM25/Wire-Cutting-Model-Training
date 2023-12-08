import numpy as np
from diagram import diagram
import argparse
import os

#Usage: python3 generate_data.py --N --dest
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='In this script we will essentially create all the training data and store their labels')
    parser.add_argument('--N',help= 'Number of diagrams we want to generate')
    parser.add_argument('--dest', help='folder where we will store all the data')
    args = parser.parse_args()

    matrices = []
    labels = []
    for _ in range(int(args.N)):
        dgrm = diagram.createDiagram()
        danger = np.array(dgrm.layWires())
        matrices.append(dgrm.data.flatten())
        labels.append(danger)
    matrices = np.array(matrices)
    labels = np.array(labels)

    # In case the necessary directory hasn't been made yet
    os.makedirs(args.dest, exist_ok=True)
    np.save(arr=matrices,file=f'{args.dest}/data.npy',allow_pickle=True)
    np.save(arr=labels,file=f'{args.dest}/labels.npy',allow_pickle=True)


