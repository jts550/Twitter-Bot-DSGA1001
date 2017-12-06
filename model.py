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
            self.grid_output = None
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
        #Parameters for iteration use
        self.parameters = {}
        #Set iteration parameters automatically
        self.iterationTesting()
    
    def iterationTesting(self, cv = 3, jobs = 4, update_diff = 0.02, diff = 0.001, step = 1.5, cutoff = 0.01):
        #Values for use in iteration Testing
        self.cv = cv
        self.jobs = jobs
        self.update_diff = update_diff
        self.diff = diff
        self.step = step
        self.cutoff = cutoff
        
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
        self.iterations[name].fpr, self.iterations[name].tpr, self.iterations[name].thresholds = \
            mt.roc_curve(self.y_test, self.iterations[name].predictions[:,1])
        self.iterations[name].auc = mt.auc(self.iterations[name].fpr, self.iterations[name].tpr)
        
    def plotLearnCurveMultiParam(self, iteration):
        #plot learning curve for two parameters
        
        fig, axes = plt.subplots(figsize=(8,6))
        data = pd.pivot_table(iteration.grid_output, index=iteration.params[0], columns=iteration.params[1])
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
        grid = iteration.grid_output
        #Main Line
        x_val = grid[iteration.params]
        max_idx = grid.mean_score.argmax()
        axes.plot(x_val, grid.mean_score, 'C0', label='Mean AUC')
        #Std Error lines
        axes.plot(x_val, grid.lbound, 'C0', label='-1 Std.Err', linestyle='--')
        axes.plot(x_val, grid.ubound, 'C0', label='+1 Std.Err', linestyle='--')
        #1 std error rule
        error_rule = grid.mean_score.max() - grid.std_err[max_idx]
        xmin = x_val.min()
        xmax = x_val.max()
        plt.hlines(xmin = xmin, xmax = xmax, y = error_rule, color = 'r')
        #Plot aesthetics
        plt.legend()
        plt.tight_layout()
        plt.title("Learning Curve of {0}:\n{1}".format(iteration.name, str(iteration.params)))
        
    def regularizedParamaterSearch(self, iteration, param, param_clean, grid, is_int, des):
        #Find range for parameter, return mean
        max_idx = grid.mean_score.argmax()
        #run until there is a non-empty df
        cut = True
        diff = self.diff
        while(cut):
            if diff >= self.cutoff:
                #check if parameter is int
                if is_int:
                    self.parameters[param_clean] = int(grid[param][max_idx])
                else:
                    self.parameters[param_clean] = grid[param][max_idx]
                return
            stripped = grid
            #cut down based on ascending 
            if des:
                stripped = stripped[stripped[param] < grid[param][max_idx]]
            else:
                stripped = stripped[stripped[param] > grid[param][max_idx]]
            #Cut down by bounds
            further = stripped[(abs(stripped.mean_score - grid.mean_score.max()) <= diff) & 
                               (abs(stripped.lbound - grid.lbound[max_idx] <= diff)) &
                               (abs(stripped.ubound - grid.ubound[max_idx] <= diff))]
            #Stop loop, if df is non-empty
            cut = further.empty
            diff *=  self.step
        if is_int:
            self.parameters[param_clean] = int(further[param].mean())
        else:
            self.parameters[param_clean] = further[param].mean()
        
    def regularizedParameters(self, iteration, des):
        #update for regularized parameters for next run through
        grid =  iteration.grid_output
        
        #Run through parameters
        for param in iteration.params:
            #Check if int
            param_clean = re.sub('estimator__', '',param)
            is_int = isinstance(self.estimator_step[param_clean], int)
            #Find value
            self.regularizedParamaterSearch(iteration, param, param_clean, grid, is_int, des)
            
    def bestParameters(self, iteration):
        #update parameters for next run through
        grid = iteration.grid_output
        #Run thorugh parameters
        for param in iteration.params:
            #Check if int
            param_clean = re.sub('estimator__', '',param)
            is_int = isinstance(self.estimator_step[param_clean], int)
            #update with best
            if is_int:
                self.parameters[param_clean] = int(grid[param][grid.mean_score.idxmax()])
            else:
                self.parameters[param_clean] = grid[param][grid.mean_score.idxmax()]
                    
    def updateParameters(self, iteration, reg, des, not_default):
        #check if default pipeline
        if not(not_default):
            estimator_index = list([x[0] for x in iteration.model.steps]).index('estimator')
            self.estimator_step = iteration.model.steps[estimator_index][1].get_params()
            #Check if regularized or best param
            if reg:
                self.regularizedParameters(iteration, des)
            else:
                self.bestParameters(iteration)
        #With different pipe, set params by hand
        else:
            return
        
    def bestIteration(self, iteration, reg, des, not_default):
        #update best iteration
        #update the parameters to use for other iterations
        if abs(iteration.auc - self.best_iteration.auc) <= self.update_diff:
            self.updateParameters(iteration, reg, des, not_default)
        #only update if better
        if iteration.auc > self.best_iteration.auc: 
            self.best_iteration = iteration
        
        

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
        scores_columns = ["split" + str(x) + "_test_score" for x in range(0,iteration.grid.cv)]
        std_err = np.sqrt(grid_summary[scores_columns].var(axis = 1, ddof = 0)/iteration.grid.cv)
        grid_new.insert(grid_new.columns.get_loc("mean_score")+1, 'std_err', std_err)
        #CI lines
        grid_new['lbound'] = grid_new.mean_score - grid_new.std_err
        grid_new['ubound'] = grid_new.mean_score + grid_new.std_err
        
        self.iterations[name].grid_output = grid_new
        
    def buildPipeline(self, name, pipeline, estimator):
        #Build pipeline, indicator if default pipeline
        if pipeline:
            self.iterations[name].pipeline = pipeline
            not_default = True
        else:
            self.iterations[name].pipeline = Pipeline([('estimator', estimator)])
            not_default = False
        return self.iterations[name].pipeline, not_default

    def addIteration(self, name, estimator, param_grid, pipeline = None, reg = False, des = True, plot = True):
        #BUild out gridsearch, cv model and plot learning curve
        self.iterations[name] = self.Iteration(name)
        #Build pipeline 
        pipeline, not_default = self.buildPipeline(name, pipeline, estimator)
        #Build CV grid then run on data
        self.iterations[name].grid = \
            ms.GridSearchCV(pipeline, param_grid = param_grid, cv = self.cv, scoring = "roc_auc", return_train_score=False)
        self.iterations[name].grid.fit(self.X_train, self.y_train)
        #print results
        print("Best Score: {:0.6}\n".format(self.iterations[name].grid.best_score_))
        print("Best Params: ",self.iterations[name].grid.best_params_)
        #Save parameters
        self.iterations[name].params = list(param_grid.keys())
        self.gridSearchSummary(self.iterations[name], name)
        self.iterations[name].model = self.iterations[name].grid.best_estimator_
        #Predictions and best iteration
        self.predict(name)
        self.bestIteration(self.iterations[name], reg, des, not_default)
        #plot learning curve if wanted
        if len(self.iterations[name].params) == 1 & plot:
            self.plotLearnCurve(self.iterations[name])
        elif plot:
            self.plotLearnCurveMultiParam(self.iterations[name])
        
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

