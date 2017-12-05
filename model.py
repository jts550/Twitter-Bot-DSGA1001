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
        
    def plotLearnCurveMultiParam(self, iteration):
        #plot learning curve for two parameters
        
        fig, axes = plt.subplots(figsize=(8,6))
        data = pd.pivot_table(iteration.grid_summary, index=iteration.params[0], columns=iteration.params[1])
        #Main line
        data['mean_score'].plot(ax=axes, figsize=(8,6))
        #std error lines
        (data['lbound']).plot(ax=axes, figsize=(8,6),alpha=0.25,linestyle='--')
        (data['ubound']).plot(ax=axes, figsize=(8,6),alpha=0.25,linestyle='--')
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
        plt.title("Learning Curve of {0}:\n{1}".format(iteration.name, str(iteration.params)))
        
    def plotLearnCurve(self, iteration):
        #plot learning curve for one parameter
        fig, axes = plt.subplots(figsize=(8,6))
        #Main Line
        x_val = iteration.grid_summary[iteration.params]
        max_idx = iteration.grid_summary.mean_score.argmax()
        axes.plot(x_val, iteration.grid_summary.mean_score, 'C0', label='Mean AUC')
        #Std Error lines
        axes.plot(x_val, iteration.grid_summary.lbound, 'C0', label='-1 Std.Err', linestyle='--')
        axes.plot(x_val, iteration.grid_summary.ubound, 'C0', label='+1 Std.Err', linestyle='--')
        #1 std error rule
        error_rule = iteration.grid_summary.mean_score.max() - iteration.grid_summary.std_err[max_idx]
        xmin = x_val.min()
        xmax = x_val.max()
        plt.hlines(xmin = xmin, xmax = xmax, y = error_rule, color = 'r')
        #Plot aesthetics
        plt.legend()
        plt.tight_layout()
        plt.title("Learning Curve of {0}:\n{1}".format(iteration.name, str(iteration.params)))
        
    def regularizedParamSearch(self, iteration, param, grid, descending, diff):
        #1 std error rule
        max_idx = grid.mean_score.argmax()
        param_clean = re.sub('estimator__', '',param)
        steps = iteration.model.steps
        param_index = list([x[0] for x in steps]).index('estimator')
        is_int = isinstance(steps[param_index][1].get_params()[param_clean], int)
        #within 1 std error rule bound
        cut = True
        diff = diff
        while(cut):
            if diff >= 0.1:
                #check if parameter is int
                if is_int:
                    self.parameters[param_clean] = int(grid[param][max_idx])
                else:
                    self.parameters[param_clean] = grid[param][max_idx]
                return
            stripped = grid
            stripped = stripped[(abs(stripped.mean_score - grid.mean_score.max()) <= diff) & 
                                (abs(stripped.lbound - grid.lbound[max_idx] <= diff))]
           #cut down based on ascending parameter
            if descending:
                further = stripped[stripped[param] <= grid[param][max_idx]]
            else:
                further = stripped[stripped[param] >= grid[param][max_idx]]
            cut = further.empty
            diff *=  10
        if is_int:
            self.parameters[param_clean] = int(further[param].mean())
        else:
            self.parameters[param_clean] = further[param].mean()
        
    def regularizedParams(self, iteration, descending, diff):
        #update for regularized parameters for next run through
        grid =  iteration.grid_summary
        #go through params
        for param in iteration.params:
            self.regularizedParamSearch(iteration, param, grid, descending, diff)
            
    def bestParams(self, iteration):
        #update parameters for next run through
        grid = iteration.grid_summary
        for param in iteration.params:
                param_clean = re.sub('estimator__', '',param)
                steps = iteration.model.steps
                param_index = list([x[0] for x in steps]).index('estimator')
                is_int = isinstance(steps[param_index][1].get_params()[param_clean], int)
                #update with best
                if is_int:
                    self.parameters[param_clean] = int(grid[param][grid.mean_score.idxmax()])
                else:
                    self.parameters[param_clean] = grid[param][grid.mean_score.idxmax()]
        
    def bestIteration(self, iteration, descending, diff, regularized, not_default):
        #update best cv-model
        #only update if better
        if iteration.auc > self.best_iteration.auc: 
            self.best_iteration = iteration
            #update the paramters to use for other iterations
            if not(not_default):
                if regularized:
                    self.regularizedParams(iteration, descending, diff)
                else:
                    self.bestParams(iteration)
            else:
                return

    def gridSearchSummary(self, iteration, name):
        #get data from gridsearch and return in better format
        grid_summary = pd.DataFrame(iteration.grid.cv_results_)
        #iterate through parameters
        grid_new = defaultdict(list)
        for row in grid_summary.params:
            for key, value in row.items():
                grid_new[key] += [value]
        grid_new = pd.DataFrame(grid_new)
        #mean score
        grid_new['mean_score'] = grid_summary.mean_test_score
        #std error
        scores_columns = ["split" + str(x)+ "_test_score" for x in range(0,iteration.grid.cv)]
        std_err = np.sqrt(grid_summary[scores_columns].var(axis = 1, ddof = 0)/iteration.grid.cv)
        grid_new.insert(grid_new.columns.get_loc("mean_score")+1, 'std_err', std_err)
        #CI lines
        grid_new['lbound'] = grid_new.mean_score - grid_new.std_err
        grid_new['ubound'] = grid_new.mean_score + grid_new.std_err
        
        self.iterations[name].grid_summary = grid_new

    def addIteration(self, name, estimator, param_grid, default = None, descending = True, diff = 0.01, regularized = False, plot = True):
        #BUild out gridsearch, cv model and plot learning curve
        self.iterations[name] = self.Iteration(name)
        #Build pipeline 
        if default:
            self.iterations[name].pipeline = default
            not_default = True
        else:
            self.iterations[name].pipeline = Pipeline([('variance_thresh', fs.VarianceThreshold()), ('estimator', estimator)])
            not_default = False
        #Build CV grid then run on data
        self.iterations[name].grid = \
            ms.GridSearchCV(self.iterations[name].pipeline, param_grid = param_grid, cv = 5, scoring = "roc_auc", n_jobs = 4)
        self.iterations[name].grid.fit(self.X_train, self.y_train)
        #print results
        print("Best Score: {:0.6}\n".format(self.iterations[name].grid.best_score_))
        print("Best Params: ",self.iterations[name].grid.best_params_)
        #add summary and best model to Model
        self.iterations[name].params = list(param_grid.keys())
        self.gridSearchSummary(self.iterations[name], name)
        self.iterations[name].model = self.iterations[name].grid.best_estimator_
        self.predict(name)
        self.bestIteration(self.iterations[name], descending, diff, regularized, not_default)
        #plot learning curve if wanted
        if len(self.iterations[name].params) == 1 & plot:
            self.plotLearnCurve(self.iterations[name])
        elif plot:
            self.plotLearnCurveMultiParam(self.iterations[name])
            
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

