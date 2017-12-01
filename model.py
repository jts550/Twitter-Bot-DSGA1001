import pandas as pd
import numpy as np
import re

from sklearn import metrics as mt
from sklearn import preprocessing as pre
from sklearn import model_selection as ms
from sklearn import feature_selection as fs
from sklearn.pipeline import Pipeline

from collections import defaultdict
from itertools import chain

import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

#Model class
class Model:
    
    class Iteration:
        
        def __init__(self, name):
            self.name = name
            self.params = None
            self.grid = None
            self.grid_summary = None
            self.model = None
            self.predictions = None
    
    def __init__(self, name, model, X_train, y_train, X_test, y_test):
        #Name 
        self.name = name
        self.iterations = {}
        self.best_iteration = None
        #Data            
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.baseline(model)
        #
        self.parameters = {}
        
    def baseline(self, model):
        #Initialzie Baseline Model
        self.iterations["Baseline"] = self.Iteration("Baseline")
        self.iterations["Baseline"].model = model
        self.iterations["Baseline"].model.fit(self.X_train,self.y_train)
        #SET PARAMS FOR BASELINE
        self.predict("Baseline")
        self.best_iteration = self.iterations["Baseline"]
        
    def predict(self, name):
        #cv models predictions
        self.iterations[name].predictions = self.iterations[name].model.predict_proba(self.X_test)
        #auc
        self.iterations[name].fpr, self.iterations[name].tpr, self.iterations[name].threshholds = \
            mt.roc_curve(self.y_test, self.iterations[name].predictions[:,1])
        self.iterations[name].auc = mt.auc(self.iterations[name].fpr, self.iterations[name].tpr)
        
    def bestIteration(self, name):
        #update best cv-model
        model = self.iterations[name]
        #only update if better
        if model.auc > self.best_iteration.auc: 
            self.best_iteration = self.iterations[name]
            #update the paramters to use for other iterations
            for param in model.params:
                #Get the param
                param_clean = re.sub('estimator__', '',param)
                #update with best
                self.parameters[param_clean] = model.grid_summary[param][model.grid_summary.mean_score.idxmax()]
            
    def plotLearnCurveMultiParam(self, name):
        #plot learning curve for two parameters
        
        model = self.iterations[name]
        
        fig, axes = plt.subplots(figsize=(8,6))
        data = pd.pivot_table(model.grid_summary, index=model.params[0], columns=model.params[1])
        #Main line
        data['mean_score'].plot(ax=axes, figsize=(8,6))
        #std error lines
        (data['Lbound']).plot(ax=axes, figsize=(8,6),alpha=0.25,linestyle='--')
        (data['Ubound']).plot(ax=axes, figsize=(8,6),alpha=0.25,linestyle='--')
        #1 std error rule
        c_max = data['mean_score'].max().idxmax()
        r_max = data['mean_score'][c_max].idxmax()
        error_rule = data['mean_score'].loc[r_max, c_max] - data['std_err'].loc[r_max, c_max]
        xmin = data.index.values.min()
        xmax = data.index.values.max()
        plt.hlines(xmin = xmin, xmax = xmax, y = error_rule, color = 'r')  
        #Plot aesthetics
        plt.legend()
        plt.tight_layout()
        plt.title("Learning Curve of {0}:\n{1}".format(model.name, str(model.params)))
        
    def plotLearnCurve(self, name):
        #plot learning curve for one parameter
        model = self.iterations[name]
        fig, axes = plt.subplots(figsize=(8,6))
        #Main Line
        x_val = model.grid_summary[model.params]
        max_idx = model.grid_summary.mean_score.argmax()
        axes.plot(x_val, model.grid_summary.mean_score, 'C0', label='Mean AUC')
        #Std Error lines
        axes.plot(x_val, model.grid_summary.Lbound, 'C0', label='-1 Std.Err', linestyle='--')
        axes.plot(x_val, model.grid_summary.Ubound, 'C0', label='+1 Std.Err', linestyle='--')
        #1 std error rule
        error_rule = model.grid_summary.mean_score.max() - model.grid_summary.std_err[max_idx]
        xmin = x_val.min()
        xmax = x_val.max()
        plt.hlines(xmin = xmin, xmax = xmax, y = error_rule, color = 'r')
        #Plot aesthetics
        plt.legend()
        plt.tight_layout()
        plt.title("Learning Curve of {0}:\n{1}".format(model.name, str(model.params)))

    def gridSearchSummary(self, name):
        #get data from gridsearch and return in better format
        model = self.iterations[name]
        grid_summary = pd.DataFrame(model.grid.cv_results_)
        #iterate through parameters
        grid_new = defaultdict(list)
        for row in grid_summary.params:
            for key, value in row.items():
                grid_new[key] += [value]
        grid_new = pd.DataFrame(grid_new)
        #mean score
        grid_new['mean_score'] = grid_summary.mean_test_score
        #std error
        scores_columns = ["split" + str(x)+ "_test_score" for x in range(0,model.grid.cv)]
        std_err = np.sqrt(grid_summary[scores_columns].var(axis = 1, ddof = 0)/model.grid.cv)
        grid_new.insert(grid_new.columns.get_loc("mean_score")+1, 'std_err', std_err)
        #CI lines
        grid_new['Lbound'] = grid_new.mean_score - grid_new.std_err
        grid_new['Ubound'] = grid_new.mean_score + grid_new.std_err
        
        return grid_new

    def addIteration(self, name, estimator, param_grid, cv = 5, scoring = "roc_auc", plot = True):
        #BUild out gridsearch, cv model and plot learning curve
        self.iterations[name] = self.Iteration(name)
        #Build pipeline 
        self.iterations[name].pipeline = Pipeline([('variance_thresh', fs.VarianceThreshold()), ('estimator', estimator)])
        #Build CV grid then run on data
        self.iterations[name].grid = \
            ms.GridSearchCV(self.iterations[name].pipeline,param_grid = param_grid, cv = cv, scoring = scoring, n_jobs = 4)
        self.iterations[name].grid.fit(self.X_train, self.y_train)
        #print results
        print("Best Score: {:0.6}\n".format(self.iterations[name].grid.best_score_))
        print("Best Params: ",self.iterations[name].grid.best_params_)
        #add summary and best model to Model
        self.iterations[name].params = list(param_grid.keys())
        self.iterations[name].grid_summary = self.gridSearchSummary(name)
        self.iterations[name].model = self.iterations[name].grid.best_estimator_
        self.predict(name)
        self.bestIteration(name)
        #plot learning curve if wanted
        if len(self.iterations[name].params) == 1 & plot:
            self.plotLearnCurve(name)
        elif plot:
            self.plotLearnCurveMultiParam(name)
            
    def plotAUC(self, name):
        #Plot auc of model
        #check if cross-validated model
        fpr = self.iterations[name].fpr
        tpr = self.iterations[name].tpr
        auc = self.iterations[name].auc
        title = "ROC Curve {}".format(name)
        #plot auc
        fig, axes = plt.subplots(1,1, figsize=(8,6))
        axes.plot(fpr, tpr, label = " (AUC = {:0.3})".format(roc_auc)) 
        #plot aesthetics
        plt.title(title)
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.legend()
        
    def withinCompare(self):
    #plot aucs of models
        fig, axes = plt.subplots(1,1, figsize=(8,6))
        for name in self.iterations:
            fpr = self.iterations[name].fpr
            tpr = self.iterations[name].tpr
            auc = self.iterations[name].auc
            axes.plot(fpr, tpr, label = name + " (AUC = {:0.3})".format(auc))
        #plot aesthetics
        plt.title("ROC Curves")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.legend()          

