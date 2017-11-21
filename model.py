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
    results = None
    predictions = None
    pipeline = None 
    cv_models = {}
    best_cv = None
    
    class CV_Model:
        params = None
        grid = None
        grid_summary = None
        model = None
        predictions = None
        
        def __init__(self, name):
            self.name = name
    
    def __init__(self, name, model, X_train, y_train, X_test, y_test):
        #Name 
        self.name = name
        #Data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        #Initialzie Baseline Model
        self.model = model
        self.fit = self.model.fit(X_train,y_train)
        self.predict()
        
    def predict(self):
        #baseline model predictions
        self.predictions = self.model.predict_proba(self.X_test)
        fpr, tpr, thresholds = mt.roc_curve(self.y_test, self.predictions[:,1])
        self.auc = mt.auc(fpr, tpr)
        
    def cvPredict(self, name):
        #cv models predictions
        self.cv_models[name].predictions = self.cv_models[name].cv_model.predict_proba(self.X_test)
        fpr, tpr, thresholds = mt.roc_curve(self.y_test, self.cv_models[name].predictions[:,1])
        self.cv_models[name].auc = mt.auc(fpr, tpr)
        
    def bestCVModel(self, name):
        if self.best_cv is None:
            self.best_cv = name
        elif self.cv_models[name].auc > self.cv_models[self.best_cv].auc: 
            self.best_cv = name
            
    def plotLearnCurveMultiParam(self, name):
        #plot learning curve for two parameters
        fig, axes = plt.subplots(figsize=(8,6))
        data = pd.pivot_table(self.cv_models[name].grid_summary, 
                              index=self.cv_models[name].params[0], columns=self.cv_models[name].params[1])
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
        plt.title("Learning Curve of {0}:\n{1}".format(self.cv_models[name].name, str(self.cv_models[name].params)))
        
    def plotLearnCurve(self, name):
        #plot learning curve for one parameter
        fig, axes = plt.subplots(figsize=(8,6))
        #Main Line
        x_val = self.cv_models[name].grid_summary[self.cv_models[name].params]
        max_idx = self.cv_models[name].grid_summary.mean_score.argmax()
        axes.plot(x_val, self.cv_models[name].grid_summary.mean_score, 'C0', label='Mean AUC')
        #Std Error lines
        lower = self.cv_models[name].grid_summary.mean_score - self.cv_models[name].grid_summary.std_err
        axes.plot(x_val, lower, 'C0', label='-1 Std.Err', linestyle='--')
        upper = self.cv_models[name].grid_summary.mean_score + self.cv_models[name].grid_summary.std_err
        axes.plot(x_val, upper, 'C0', label='+1 Std.Err', linestyle='--')
        #1 std error rule
        error_rule = self.cv_models[name].grid_summary.mean_score.max() - self.cv_models[name].grid_summary.std_err[max_idx]
        xmin = x_val.min()
        xmax = x_val.max()
        plt.hlines(xmin = xmin, xmax = xmax, y = error_rule, color = 'r')
        #Plot aesthetics
        plt.legend()
        plt.tight_layout()
        plt.title("Learning Curve of {0}:\n{1}".format(self.cv_models[name].name, str(self.cv_models[name].params)))

    def gridSearchSummary(self, name):
        #get data from gridsearch and return in better format
        grid_summary = pd.DataFrame(self.cv_models[name].grid.cv_results_)
        #iterate through parameters
        params_summary = defaultdict(list)
        for row in grid_summary.params:
            for key, value in row.items():
                params_summary[key] += [value]
        params_summary_df = pd.DataFrame(params_summary)
        #mean score
        params_summary_df['mean_score'] = grid_summary.mean_test_score
        #std error
        scores_columns = ["split" + str(x)+ "_test_score" for x in range(0,self.cv_models[name].grid.cv)]
        std_err = np.sqrt(grid_summary[scores_columns].var(axis = 1, ddof = 0)/self.cv_models[name].grid.cv)
        params_summary_df.insert(params_summary_df.columns.get_loc("mean_score")+1, 'std_err', std_err)

        return params_summary_df

    def tuningIteration(self, name, estimator, param_grid, cv = 5, scoring = "roc_auc", plot = True):
        #BUild out gridsearch, cv model and plot learning curve
        #Build pipeline 
        self.cv_models[name] = self.CV_Model(name)
        self.cv_models[name].pipeline = Pipeline([('variance_thresh', fs.VarianceThreshold()), ('estimator', estimator)])
        #Build CV grid then run on data
        self.cv_models[name].grid = \
            ms.GridSearchCV(self.cv_models[name].pipeline,param_grid = param_grid, cv = cv, scoring = scoring, n_jobs = 4)
        self.cv_models[name].grid.fit(self.X_train, self.y_train)
        #print results
        print("Best Score: {:0.6}\n".format(self.cv_models[name].grid.best_score_))
        print("Best Params: ",self.cv_models[name].grid.best_params_)
        #add summary and best model to Model
        self.cv_models[name].params = list(param_grid.keys())
        self.cv_models[name].grid_summary = self.gridSearchSummary(name)
        self.cv_models[name].cv_model = self.cv_models[name].grid.best_estimator_
        self.cvPredict(name)
        self.bestCVModel(name)
        #plot learning curve if wanted
        if len(self.cv_models[name].params) == 1 & plot:
            self.plotLearnCurve(name)
        elif plot:
            self.plotLearnCurveMultiParam(name)
            
    def plotAUC(self, X_test, y_test, name):
        #Plot auc of model
        #check if cross-validated model
        if name:
            predictions = self.cv_models[name].predictions
        else:
            predictions = self.predictions
        #plot auc
        fig, axes = plt.subplots(1,1, figsize=(8,6))
        fpr, tpr, thresholds = mt.roc_curve(y_test, predictions[:,1])
        roc_auc = mt.auc(fpr, tpr)
        axes.plot(fpr, tpr, label = " (AUC = {:0.3})".format(roc_auc)) 
        #plot aesthetics
        plt.title("ROC Curve {}".format(self.cv_models[name].name))
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.legend()
            
    def compareModels(self):
        #compare auc of baseline model and cv-models
        #Get all predictions and models
        predictions = [[self.predictions]]
        predictions.append([value.predictions for key, value in self.cv_models.items()])
        predictions = list(chain(*predictions))
        
        names = [[self.name]]
        names.append([value.name for key, value in self.cv_models.items()])
        names = list(chain(*names))
        
        preds_zip = zip(predictions,names)

        fig, axes = plt.subplots(1,1, figsize=(8,6))
        for preds, name in preds_zip:
            fpr, tpr, thresholds = mt.roc_curve(self.y_test, preds[:,1])
            roc_auc = mt.auc(fpr, tpr)
            axes.plot(fpr, tpr, label = name + " (AUC = {:0.3})".format(roc_auc))
        #plot aesthetics
        plt.title("ROC Curves")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.legend()