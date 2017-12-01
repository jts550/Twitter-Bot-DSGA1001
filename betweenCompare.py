import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics as mt

def betweenCompare(models, axes = None):
    if axes:
        pass
    else:
        fig, axes = plt.subplots(1,1, figsize=(8,6))
    for it in models:
        model = it.best_iteration
        axes.plot(model.fpr, model.tpr, label = it.name + " (AUC = {:0.3})".format(model.auc))
    #plot aesthetics
    plt.title("ROC Curves")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend()