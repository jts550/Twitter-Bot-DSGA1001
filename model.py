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

from numba import jitclass, jit, generated_jit, void

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
            self.pipeline = None

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

    def iterationTesting(self, cv = 5, jobs = 4, update_diff = 0.02, diff = 0.001, step = 1.5, cutoff = 0.01):
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
        self.iterations[name].predictions = self.iterations[name].model.predict_proba(self.X_test)[:,1]
        #auc
        self.iterations[name].fpr, self.iterations[name].tpr, self.iterations[name].thresholds = \
            mt.roc_curve(self.y_test, self.iterations[name].predictions)
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

    def isInt(self, iteration = None, param = None, param_clean = None, step = False):
        #check if parameter is int, split between pipeline and classifier
        if step:
            return isinstance(step[1].get_params()[param_clean], int)
        else: 
            return isinstance(iteration.model.get_params()[param], int)
   
    def regularizedParamaterSearch(self, iteration, grid, param, des):
        #Find range for parameter, return mean
        max_idx = grid.mean_score.argmax()
        #regularization direction
        if des:
            grid_cut = grid[grid[param] < grid[param][max_idx]]
        else:
            grid_cut = grid[grid[param] > grid[param][max_idx]]
        #run until there is a non-empty df
        cut = True
        diff = self.diff
        while(cut):
            if diff >= self.cutoff:
                return grid[param][max_idx]
            grid_cut2 = grid_cut            
            #Cut down by bounds
            grid_cut3 = grid_cut2[(abs(grid_cut2.mean_score - grid.mean_score.max()) <= diff) & 
                                  (abs(grid_cut2.lbound - grid.lbound[max_idx] <= diff)) &
                                  (abs(grid_cut2.ubound - grid.ubound[max_idx] <= diff))]
            #Stop loop, if df is non-empty
            cut = grid_cut3.empty
            diff *=  self.step
        return grid_cut3[param].mean()
    
    def bestParametersSearch(self, iteration, grid, param): 
        #return the value of the best 
        return grid[param][grid.mean_score.idxmax()]
    
    def parametersPipelineUpdate(self, iteration, grid, params, reg, des):
        #Update for pipeline
        #Go through steps
        steps = iteration.model.steps
        
        for step in steps:
            step_dict = {}   
            extra = "{}__".format(step[0])
            #Go through parameters
            for param in params:
                if(extra in param):
                    #regularized or best param
                    if reg:
                        value = self.regularizedParamaterSearch(iteration, grid, param, des)
                    else:
                        value = self.bestParametersSearch(iteration, grid, param)
                    #if param is in int
                    if self.isInt(param_clean = re.sub(extra, '',param), step = step):
                        value = int(value)
                    step_dict[re.sub(extra, '',param)] = value
            
            if step[0] not in self.parameters:
                self.parameters[step[0]] = step_dict
            else:
                self.parameters[step[0]].update(step_dict)            

    def parametersUpdate(self, iteration, grid, params, reg, des):
        #update the parameters of a normal sklearn classifer
        for param in params:
            #regularized or best param
            if reg:
                value = self.regularizedParamaterSearch(iteration, grid, param, des)
            else:
                value = self.bestParametersSearch(iteration, grid, param)
            #param is int
            if self.isInt(iteration = iteration, param = param):
                value = int(value)
            self.parameters[param] = value                    

    def updateParameters(self, iteration, reg, des):
        #check if default pipeline
        grid = iteration.grid_output
        params = iteration.params
        
        if iteration.pipeline:
            self.parametersPipelineUpdate(iteration, grid, params, reg, des)                
        else:
            self.parametersUpdate(iteration, grid, params, reg, des)

    def bestIteration(self, iteration, reg, des):
        #update best iteration
        #update the parameters to use for other iterations
        #baseline has no grid
        if (self.best_iteration.name == "Baseline"):
            if(abs(iteration.auc - self.best_iteration.auc) <= self.update_diff):
                self.updateParameters(iteration, reg, des)
        else:
            max_idx = self.best_iteration.grid_output.mean_score.idxmax()
            if(iteration.auc >  self.best_iteration.grid_output.lbound[max_idx]):
                self.updateParameters(iteration, reg, des)
        #only update if better
        if iteration.auc > self.best_iteration.auc: 
            self.best_iteration = iteration
    
    def gridSearchSummary(self, iteration, name):
        #get data from gridsearch and return in better format
        grid_summary = iteration.grid.cv_results_
        n = iteration.grid.cv
        #mean score
        grid_new = pd.DataFrame(grid_summary['params'])
        grid_new['mean_score'] = grid_summary['mean_test_score']
        #std errort
        scores_columns = ["split{}_test_score".format(x) for x in np.arange(n)]
        grid_new['std_err'] = np.sqrt(pd.DataFrame(grid_summary)[scores_columns].var(axis = 1, ddof = 0)/n)
        #CI lines
        grid_new['lbound'] = grid_new.mean_score - grid_new.std_err
        grid_new['ubound'] = grid_new.mean_score + grid_new.std_err
        
        self.iterations[name].grid_output = grid_new

    def addIteration(self, name, estimator, param_grid, reg = False, des = True, plot = True):
        #BUild out gridsearch, cv model and plot learning curve
        self.iterations[name] = self.Iteration(name)
        #Build pipeline 
        self.iterations[name].pipeline = isinstance(estimator, Pipeline)
        #Build CV grid then run on data
        self.iterations[name].grid = \
            ms.GridSearchCV(estimator, param_grid = param_grid, cv = self.cv, scoring = "roc_auc", return_train_score = False)
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
        self.bestIteration(self.iterations[name], reg, des)
        #plot learning curve if wanted
        if len(self.iterations[name].params) == 1 & plot:
            self.plotLearnCurve(self.iterations[name])
        elif plot:
            self.plotLearnCurveMultiParam(self.iterations[name] )

    def withinCompare(self):
    #plot aucs of models
        fig, axes = plt.subplots(1,1, figsize=(8,6))
        for name in self.iterations.keys():
            fpr = self.iterations[name].fpr
            tpr = self.iterations[name].tpr
            auc = self.iterations[name].auc
            axes.plot(fpr, tpr, label = "{} (AUC = {:0.3})".format(self.iterations[name].name, auc))
        #plot aesthetics
        plt.title("ROC Curves")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.legend()          

