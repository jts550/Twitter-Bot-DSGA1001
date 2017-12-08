import pandas as pd
import numpy as np

import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from betweenCompare import betweenCompare

import locale
locale.setlocale( locale.LC_ALL, '' )

def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x*1e-6)

def to_percent(x, pos):
    'The two args are the value and tick position'
    return '%.0f%%' % (x*100)
    
def expectedValue(model, y_test, thresholds, cost_mat):
    # fp_cost and fn_cost should be the change in revenue associated with 1000 ad requests (i.e. rCPM change)
    # output of expected value is the expected rCPM 
    #predictions
    class_preds = [1 if x else 0 for x in model.predictions > thresholds]
    #confusion matrix
    conf_mat = confusion_matrix(y_test, class_preds)/len(model.predictions)
    #expected value
    ev = conf_mat[0,1]*cost_mat[0,1] + conf_mat[1,0]*cost_mat[1,0]
    
    return ev

def dataProfit(timeframe, rcpm_change, pct_instance):
    #get profit data for each pct instance and rcpm change
    #adjust timeframe for yearly and monthly
    if timeframe=='yearly':
        profit = list(map(lambda x: x/1000*12*450*10**9, rcpm_change))
    elif timeframe=='monthly':
        profit = list(map(lambda x: x/1000*450*10**9, rcpm_change))
    else:
        raise ValueError('timeframe must be "yearly" or "monthly"')
    #profit data frame, pct_instance, profit and rcpm_change
    profitData = pd.DataFrame({'pct_instance': pct_instance, 'profit': profit, 'rcpm_change': rcpm_change})
    #sort by pct_instance
    profitData = profitData.sort_values('pct_instance')
    #percent change from smallest
    profitData['profit_change'] = profitData.profit.transform(lambda x: (x/x[0]-1)*-1)
    #difference between smallest
    profitData['profit_diff'] = profitData.profit.transform(lambda x: (x-x[0]))
    
    return profitData

def maxProfit(profitData):
    #useful values from profit data
    profits = {}
    #max idx
    max_idx = profitData.profit_change.idxmax()
    #max change
    profits['change'] = profitData.profit_change.iloc[max_idx]
    #max pct instace
    profits['pct_instance'] = profitData.pct_instance.iloc[max_idx]
    #max profit
    profits['max'] = profitData.profit.iloc[max_idx]
    #max diff
    profits['diff'] = profitData.profit_diff.iloc[max_idx]
    
    return profits  
    
def calculateProfit(model, y_test, fp_cost, fn_cost, timeframe):
    #calculate profit, pct instance and rcpm for all thresholds
    rcpm_change = []
    pct_instance = []
    #cost-benefit matrix
    cost_mat = np.array([0, fp_cost,
                    fn_cost, 0]).reshape(2,2)
    #each threshold from model
    for each_thresh in model.thresholds:
        #rcpm_change is expected value
        rcpm_change += [expectedValue(model, y_test, [each_thresh]*len(y_test), cost_mat)]
        #how many predictions are above the thresholds
        pct_instance_thresh = np.sum([model.predictions] > each_thresh)/len(y_test)    
        #add previous variable
        pct_instance += [pct_instance_thresh]
    #get data frame and dictionary
    profit = dataProfit(timeframe, rcpm_change, pct_instance)
    max_profit = maxProfit(profit)
    
    return profit, max_profit

def plotProfitChng(model, profits, max_profit, ax):
    pct_formatter = FuncFormatter(to_percent)
    #plot profit change vs pct instance
    #label with max profit change
    ax.plot(profits.pct_instance, profits.profit_change,
            label = "{} Max: {:0.1%} Pct Inst.: {:0.1%}".format(model.name,
            max_profit['change'],max_profit['pct_instance']))
    ax.yaxis.set_major_formatter(pct_formatter)
    ax.xaxis.set_major_formatter(pct_formatter)
    
def plotProfitDiff(model, profits, max_profit, ax):
    mil_formatter = FuncFormatter(millions)
    pct_formatter = FuncFormatter(to_percent)
    #plot profit change vs pct instance
    #label with max profit diff
    ax.plot(profits.pct_instance, profits.profit_change,
            label = "{} Max: {}M Pct Inst.: {:0.1%}".format(model.name,
            locale.currency(max_profit['diff']*10**-6,grouping=True),max_profit['pct_instance']))
    ax.yaxis.set_major_formatter(mil_formatter)
    ax.xaxis.set_major_formatter(pct_formatter)
            
def plotProfit(models, ax, fp_cost, fn_cost, timeframe):
    #plot change in profit
    #run through models
    for it in models:
        #get best iteration
        model = it.best_iteration
        #profits and max profit info
        profits, max_profit = calculateProfit(model, it.y_test, fp_cost, fn_cost, timeframe)
        
        #plot profit chng
        plotProfitChng(it, profits, max_profit, ax[0])
        #plot profit diff
        plotProfitDiff(it, profits, max_profit, ax[1])
    
    #set titles
    ax[0].set_title("Comparison of Profit Curves (Cost Reduction) on Test Data")
    ax[0].set_xlabel("Percentage of Test Instances")
    ax[0].set_ylabel("Expected Profit Improvement (Cost Reduction)")
    ax[0].legend()
    #set titles
    ax[1].set_title("Comparison of Profit Curves (Cost Reduction) on Test Data")
    ax[1].set_xlabel("Percentage of Test Instances")
    ax[1].set_ylabel("Expected Profit Improvement (Cost Reduction)")
    ax[1].legend()
    
def plotROC(models, ax):
    #plot model ROC curves
    pct_formatter = FuncFormatter(to_percent)
    #run through models
    for it in models:
        #get best info
        model = it.best_iteration
        ax.plot(model.fpr, model.tpr, label = "{} (AUC = {:0.3})".format(it.name, model.auc))
    #set title
    ax.set_title("Comparison of ROC Curves on Test Data")
    ax.set_xlabel("fpr")
    ax.set_ylabel("tpr")
    ax.yaxis.set_major_formatter(pct_formatter)
    ax.xaxis.set_major_formatter(pct_formatter)
    ax.legend()
    
def bootstrapAUC(model, sampsize, nruns):
    #List for holding AUC's
    aucs = []
    for i in range(nruns):
        #Split data
        samp = model.X_train.assign(target=model.y_train.values).sample(sampsize, replace = True)
        #Instantiate Model based on variable choice
        mod = model.best_iteration.model
        #Train and test model    
        mod.fit(samp.drop(['target'], 1), samp['target'])
        predictions = mod.predict_proba(model.X_test)[:,1]
        #get auc and append to list
        auc = roc_auc_score(model.y_test, predictions)
        aucs.append(auc)
    #return average, std
    return np.mean(aucs)

def plotSampleSize(models, sampsize, nruns, axes):
    x_vals = np.log2(sampsize)
    #cmap = plt.cm.get_cmap('hsv', len(models) + 1)
    #cmap(c)
    
    #Run through models
    for c, model in enumerate(models):
        #store mean  
        model_mean = []
        #run through sample sizes
        for samp in sampsize:
            #get mean and std for given model and samplesize
            mean_auc = bootstrapAUC(model, samp, nruns)
            #append 
            model_mean.append(mean_auc)
        #convert to series for plotting use
        means = pd.Series(model_mean)
    
        axes.plot(x_vals, means, label = model.name)        
    axes.set_title("Performance Plot of Sample Sizes")
    axes.set_xlabel("log(Sample Size)")
    axes.set_ylabel("AUC")
    axes.legend()
    
def plotPerformance(models, sampsize, nruns, fp_cost=-0.03, fn_cost=-0.06, timeframe='yearly'):
    # fp_cost based on Mopub rcpm
    # fn_cost = 2x fp_cost - Assume buyers act broadly because of a single bad actor
    # Twitter ad exchange request volume estimate:  
    #   https://media.mopub.com/media/filer_public/22/b5/22b58fbf-b077-4c2c-ae41-d53d06d23dd9/mopub_global_mobile_programmatic_trends_report_-_q2_2016.pdf
    
    fig = plt.figure(figsize=(14,14))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    axes = [plt.subplot(gs[0, :2], ), plt.subplot(gs[0, 2:]), plt.subplot(gs[1, :2]), plt.subplot(gs[1, 2:])]
    
    plotProfit(models, axes[:2], fp_cost, fn_cost, timeframe)
    plotROC(models, axes[2])   
    plotSampleSize(models, sampsize, nruns, axes[3])
    