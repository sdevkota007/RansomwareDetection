import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from util import getRanData, text_display

""" Sample input
filename = "./packet/CryptXXX/small_train"
"""

filename = sys.argv[1]

def main():
    global filename

    Xtrain, Ytrain = getRanData(filename, shuffle=0)
    pca = PCA()
    reduced = pca.fit_transform(Xtrain)
    
    
    # Save the PCA vectors
    try:
        output_file = filename.split('/')[-1] + ".pickle"
    except exception as e:
        print(e)
        output_file = filename + ".pickle"
        pass
    with open(output_file, "wb") as f:
        pickle.dump(reduced, f)
        pickle.dump(Xtrain, f)
        pickle.dump(Ytrain, f)


if __name__ == '__main__':
    main()
