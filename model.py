import pandas as pd
import numpy as np

from sklearn import metrics as mt
from sklearn import preprocessing as pre
from sklearn import model_selection as ms
from sklearn import feature_selection as fs
from sklearn.pipeline import Pipeline

from collections import defaultdict
from itertools import chain

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
        self.models = {}
        self.best_model = None
        #Data            
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.baseline(model)
        
    def baseline(self, model):
        #Initialzie Baseline Model
        self.models["Baseline"] = self.Iteration("Baseline")
        self.models["Baseline"].model = model
        self.models["Baseline"].model.fit(self.X_train,self.y_train)
        #SET PARAMS FOR BASELINE
        self.predict("Baseline")
        self.best_model = self.models["Baseline"]
        
    def predict(self, name):
        #cv models predictions
        self.models[name].predictions = self.models[name].model.predict_proba(self.X_test)
        #auc
        self.models[name].fpr, self.models[name].tpr, _ = mt.roc_curve(self.y_test, self.models[name].predictions[:,1])
        self.models[name].auc = mt.auc(self.models[name].fpr, self.models[name].tpr)
        
    def bestModel(self, name):
        #update best cv-model
        #only update if better
        if self.models[name].auc > self.best_model.auc: 
            self.best_model = self.models[name]
            
    def plotLearnCurveMultiParam(self, name):
        #plot learning curve for two parameters
        fig, axes = plt.subplots(figsize=(8,6))
        data = pd.pivot_table(self.models[name].grid_summary, 
                              index=self.models[name].params[0], columns=self.models[name].params[1])
        #Main line
        data['mean_score'].plot(ax=axes, figsize=(8,6))
        #std error lines
        (data['mean_score'] + data['std_err']).plot(ax=axes, figsize=(8,6),alpha=0.25,linestyle='--')
        (data['mean_score'] - data['std_err']).plot(ax=axes, figsize=(8,6),alpha=0.25,linestyle='--')
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
        plt.title("Learning Curve of {0}:\n{1}".format(self.models[name].name, str(self.models[name].params)))
        
    def plotLearnCurve(self, name):
        #plot learning curve for one parameter
        fig, axes = plt.subplots(figsize=(8,6))
        #Main Line
        x_val = self.models[name].grid_summary[self.models[name].params]
        max_idx = self.models[name].grid_summary.mean_score.argmax()
        axes.plot(x_val, self.models[name].grid_summary.mean_score, 'C0', label='Mean AUC')
        #Std Error lines
        lower = self.models[name].grid_summary.mean_score - self.models[name].grid_summary.std_err
        axes.plot(x_val, lower, 'C0', label='-1 Std.Err', linestyle='--')
        upper = self.models[name].grid_summary.mean_score + self.models[name].grid_summary.std_err
        axes.plot(x_val, upper, 'C0', label='+1 Std.Err', linestyle='--')
        #1 std error rule
        error_rule = self.models[name].grid_summary.mean_score.max() - self.models[name].grid_summary.std_err[max_idx]
        xmin = x_val.min()
        xmax = x_val.max()
        plt.hlines(xmin = xmin, xmax = xmax, y = error_rule, color = 'r')
        #Plot aesthetics
        plt.legend()
        plt.tight_layout()
        plt.title("Learning Curve of {0}:\n{1}".format(self.models[name].name, str(self.models[name].params)))

    def gridSearchSummary(self, name):
        #get data from gridsearch and return in better format
        grid_summary = pd.DataFrame(self.models[name].grid.cv_results_)
        #iterate through parameters
        params_summary = defaultdict(list)
        for row in grid_summary.params:
            for key, value in row.items():
                params_summary[key] += [value]
        params_summary_df = pd.DataFrame(params_summary)
        #mean score
        params_summary_df['mean_score'] = grid_summary.mean_test_score
        #std error
        scores_columns = ["split" + str(x)+ "_test_score" for x in range(0,self.models[name].grid.cv)]
        std_err = np.sqrt(grid_summary[scores_columns].var(axis = 1, ddof = 0)/self.models[name].grid.cv)
        params_summary_df.insert(params_summary_df.columns.get_loc("mean_score")+1, 'std_err', std_err)

        return params_summary_df

    def addIteration(self, name, estimator, param_grid, cv = 5, scoring = "roc_auc", plot = True):
        #BUild out gridsearch, cv model and plot learning curve
        self.models[name] = self.Iteration(name)
        #Build pipeline 
        self.models[name].pipeline = Pipeline([('variance_thresh', fs.VarianceThreshold()), ('estimator', estimator)])
        #Build CV grid then run on data
        self.models[name].grid = \
            ms.GridSearchCV(self.models[name].pipeline,param_grid = param_grid, cv = cv, scoring = scoring, n_jobs = 4)
        self.models[name].grid.fit(self.X_train, self.y_train)
        #print results
        print("Best Score: {:0.6}\n".format(self.models[name].grid.best_score_))
        print("Best Params: ",self.models[name].grid.best_params_)
        #add summary and best model to Model
        self.models[name].params = list(param_grid.keys())
        self.models[name].grid_summary = self.gridSearchSummary(name)
        self.models[name].model = self.models[name].grid.best_estimator_
        self.predict(name)
        self.bestModel(name)
        #plot learning curve if wanted
        if len(self.models[name].params) == 1 & plot:
            self.plotLearnCurve(name)
        elif plot:
            self.plotLearnCurveMultiParam(name)
            
    def plotAUC(self, name):
        #Plot auc of model
        #check if cross-validated model
        fpr = self.models[name].fpr
        tpr = self.models[name].tpr
        auc = self.models[name].auc
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
        for name in self.models:
            fpr = self.models[name].fpr
            tpr = self.models[name].tpr
            auc = self.models[name].auc
            axes.plot(fpr, tpr, label = name + " (AUC = {:0.3})".format(auc))
        #plot aesthetics
        plt.title("ROC Curves")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.legend()   
        