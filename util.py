from __future__ import division
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import itertools
import matplotlib.pyplot as plt


def get_seperate(train):
    """ Get seperate data form to fill in sklearn function
    @param train: train data must be in a zipped list

    @returns: datas and its labels
    """

    Xtrain = []
    Ytrain = []
    for t in train:
        Xtrain.append(t[0])
        Ytrain.append(t[1])

    Xtrain = np.matrix(Xtrain)
    return Xtrain, Ytrain

def handleErrorPacket(packetData):
    """Gets payload even if error raised.
    @param packetData: dick.

    @returns: 
    """
    

    for info in packetData.split("\n"):
        if any(method in info for method in ["GET", "POST", "PUT"]):
            return info.split("HTTP")[0]


def infoDisplay(counter, ipcounter, tcpcounter, udpcounter, httpcounter):
    """Display the information about the connection
    @param *info: those counters in the connection
    
    @returns: None
    """


    print "Total number of packets in the pcap file: ", counter
    print "Total number of ip packets: ", ipcounter
    print "Total number of tcp packets: ", tcpcounter
    print "Total number of udp packets: ", udpcounter
    print "Total number if http requests: ", httpcounter


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          plot = True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if plot:
        plt.show()
    return ax, cm


def compute_fp_fn_rate(cm):
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]

    fp_rate = fp/(tn+fp)
    fn_rate = fn/(tp+fn)
    return fp_rate, fn_rate