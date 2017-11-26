import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics as mt

def betweenCompare(models):
    fig, axes = plt.subplots(1,1, figsize=(8,6))
    for model in models:
        fpr, tpr, thresholds = mt.roc_curve(model.y_test, model.best_model.predictions[:,1])
        roc_auc = mt.auc(fpr, tpr)
        axes.plot(fpr, tpr, label = model.name + " (AUC = {:0.3})".format(roc_auc))
    #plot aesthetics
    plt.title("ROC Curves")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend()