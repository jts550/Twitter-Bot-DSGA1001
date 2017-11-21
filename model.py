import pandas as pd
import numpy as np

from sklearn import metrics as mt
from sklearn import preprocessing as pre
from sklearn import model_selection as ms
from sklearn import feature_selection as fs
from sklearn.pipeline import Pipeline

from collections import defaultdict

import matplotlib.pyplot as plt



#Model class
class Model:
    results = None
    predictions = None
    pipeline = None 
    params = None
    grid = None
    grid_summary = None
    cv_model = None
    cv_predictions = None
    
    def __init__(self, name, model, X, y):
        #initialize with name, training data, target labels, and number of target classes
        self.name = name
        self.X = X
        self.y = y
        self.model = model
        self.fit = self.model.fit(X,y)
        
    def predict(self, X):
        #baseline model predictions
        self.predictions = self.model.predict_proba(X)
        
    def cvPredict(self, X):
        #cv models predictions
        self.cv_predictions = self.cv_model.predict_proba(X)
        
    def plotLearnCurveMultiParam(self):
        #plot learning curve for two parameters
        fig, axes = plt.subplots(figsize=(8,6))
        data = pd.pivot_table(self.grid_summary, index=self.params[0], columns=self.params[1])
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
        plt.title("Learning Curve of {0}:\n{1}".format(self.name, str(self.params)))
        
    def plotLearnCurve(self):
        #plot learning curve for one parameter
        fig, axes = plt.subplots(figsize=(8,6))
        #Main Line
        x_val = self.grid_summary[self.params]
        max_idx = self.grid_summary.mean_score.argmax()
        axes.plot(x_val, self.grid_summary.mean_score, 'C0', label='Mean AUC')
        #Std Error lines
        lower = self.grid_summary.mean_score - self.grid_summary.std_err
        axes.plot(x_val, lower, 'C0', label='-1 Std.Err', linestyle='--')
        upper = self.grid_summary.mean_score + self.grid_summary.std_err
        axes.plot(x_val, upper, 'C0', label='+1 Std.Err', linestyle='--')
        #1 std error rule
        error_rule = self.grid_summary.mean_score.max() - self.grid_summary.std_err[max_idx]
        xmin = x_val.min()
        xmax = x_val.max()
        plt.hlines(xmin = xmin, xmax = xmax, y = error_rule, color = 'r')
        #Plot aesthetics
        plt.legend()
        plt.tight_layout()
        plt.title("Learning Curve of {0}:\n{1}".format(self.name, str(self.params)))

    def gridSearchSummary(self):
        #get data from gridsearch and return in better format
        grid_summary = pd.DataFrame(self.grid.cv_results_)
        #iterate through parameters
        params_summary = defaultdict(list)
        for row in grid_summary.params:
            for key, value in row.items():
                params_summary[key] += [value]
        params_summary_df = pd.DataFrame(params_summary)
        #mean score
        params_summary_df['mean_score'] = grid_summary.mean_test_score
        #std error
        scores_columns = ["split" + str(x)+ "_test_score" for x in range(0,self.grid.cv)]
        std_err = np.sqrt(grid_summary[scores_columns].var(axis = 1, ddof = 0)/self.grid.cv)
        params_summary_df.insert(params_summary_df.columns.get_loc("mean_score")+1, 'std_err', std_err)

        return params_summary_df

    def tuningIteration(self, estimator, param_grid, cv = 5, scoring = "roc_auc", plot = True):
        #BUild out gridsearch, cv model and plot learning curve
        #Build pipeline 
        self.pipeline = Pipeline([('variance_thresh', fs.VarianceThreshold()), ('estimator', estimator)])
        #Build CV grid then run on data
        self.grid = ms.GridSearchCV(self.pipeline, param_grid = param_grid, cv = cv, scoring = scoring,n_jobs = 4)
        self.grid.fit(self.X, self.y)
        #print results
        print("Best Score: {:0.6}\n".format(self.grid.best_score_))
        print("Best Params: ",self.grid.best_params_)
        #add summary and best model to Model
        self.params = list(param_grid.keys())
        self.grid_summary = self.gridSearchSummary()
        self.cv_model = self.grid.best_estimator_
        #plot learning curve if wanted
        if len(self.params) == 1 & plot:
            self.plotLearnCurve()
        elif plot:
            self.plotLearnCurveMultiParam()
            
    def plotAUC(self, X_test, y_test, cv = False):
        #Plot auc of model
        #check if cross-validated model
        if cv:
            #check existence
            if self.cv_predictions is None:
                print("Adding predictions for cross-validated model")
                self.cvPredict(X_test)
            predictions = self.cv_predictions
        else:
            #check existence
            if self.predictions is None:
                print("Adding predictions for baseline model")
                self.predict(X_test)
            predictions = self.predictions
        #plot auc
        fig, axes = plt.subplots(1,1, figsize=(8,6))
        fpr, tpr, thresholds = mt.roc_curve(y_test, predictions[:,1])
        roc_auc = mt.auc(fpr, tpr)
        axes.plot(fpr, tpr, label = " (AUC = {:0.3})".format(roc_auc)) 
        #plot aesthetics
        plt.title("ROC Curve")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.legend()
            
    def compare(self, X_test, y_test):
        #compare auc of baseline model and cv-model
        #check existence
        if self.predictions is None:
            print("Adding predictions for baseline model")
            self.predict(X_test)
        #check existence
        if self.cv_predictions is None:
            print("Adding predictions for cross-validated model")
            self.cvPredict(X_test)
        #wrapper for predictions and models
        preds_zip = zip([self.predictions, self.cv_predictions], 
                    ["Baseline {}".format(self.name), "Cross-Validated {}".format(self.name)])

        fig, axes = plt.subplots(1,1, figsize=(8,6))
        for each_preds, each_model in preds_zip:
            fpr, tpr, thresholds = mt.roc_curve(y_test, each_preds[:,1])
            roc_auc = mt.auc(fpr, tpr)
            axes.plot(fpr, tpr, label = each_model+" (AUC = {:0.3})".format(roc_auc))
        #plot aesthetics
        plt.title("ROC Curves")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.legend()
