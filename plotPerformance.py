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
    
def expectedValue(model, y_test, thresholds, fp_cost, fn_cost):
    # fp_cost and fn_cost should be the change in revenue associated with 1000 ad requests (i.e. rCPM change)
    # output of expected value is the expected rCPM 
    
    class_preds = list(map(lambda x, y: 1 if x > y else 0, model.predictions[:,1], thresholds))
    
    conf_mat = confusion_matrix(y_test, class_preds)/len(model.predictions)
    cost_mat = np.array([0, fp_cost,
                         fn_cost, 0]).reshape(2,2)
    
    ev = conf_mat[0,1]*cost_mat[0,1] + conf_mat[1,0]*cost_mat[1,0]
    
    return ev

def dataProfit(timeframe, rcpm_change, pct_instance):
    if timeframe=='yearly':
        profit = list(map(lambda x: x/1000*12*450*10**9, rcpm_change))
    elif timeframe=='monthly':
        profit = list(map(lambda x: x/1000*450*10**9, rcpm_change))
    else:
        raise ValueError('timeframe must be "yearly" or "monthly"')

    profitData = pd.DataFrame({'pct_instance': pct_instance, 'profit': profit, 'rcpm_change': rcpm_change})
    profitData = profitData.sort_values('pct_instance')
    profitData['profit_change'] = profitData.profit.transform(lambda x: (x/x[0]-1)*-1)
    profitData['profit_diff'] = profitData.profit.transform(lambda x: (x-x[0]))
    
    return profitData

def maxProfit(profit_data):
    profits = {}
    
    max_idx = profit_data.profit_change.idxmax()
    profits['change'] = profit_data.profit_change.iloc[max_idx]
    profits['pct_instance'] = profit_data.pct_instance.iloc[max_idx]
    profits['max'] = profit_data.profit.iloc[max_idx]
    profits['diff'] = profit_data.profit_diff.iloc[max_idx]
    
    return profits  
    
def calculateProfit(model, y_test, fp_cost, fn_cost, timeframe):
    
    rcpm_change = []
    pct_instance = []
    for each_thresh in model.thresholds:
        rcpm_change += [expectedValue(model, y_test, [each_thresh]*len(y_test), fp_cost, fn_cost)]
        pct_instance_thresh = np.sum(model.predictions > each_thresh)/len(y_test)    
        pct_instance += [pct_instance_thresh]
        
    profit = dataProfit(timeframe, rcpm_change, pct_instance)
    max_profits = maxProfit(profit)
    
    return profit, max_profits    

def plotProfitChange(models, ax, fp_cost, fn_cost, timeframe):
    pct_formatter = FuncFormatter(to_percent)
    
    for it in models:
        model = it.best_iteration
        profits, max_profit = calculateProfit(model, it.y_test, fp_cost, fn_cost, timeframe)

        ax.plot(profits.pct_instance, profits.profit_change,
                     label = it.name +" Max: {:0.1%} Pct Inst.: {:0.1%}".format(
                         max_profit['change'],max_profit['pct_instance']))
        ax.yaxis.set_major_formatter(pct_formatter)
        ax.xaxis.set_major_formatter(pct_formatter)

    ax.set_title("Comparison of Profit Curves (Cost Reduction) on Test Data")
    ax.set_xlabel("Percentage of Test Instances")
    ax.set_ylabel("Expected Profit Improvement (Cost Reduction)")
    ax.legend()
    

def plotProfitDiff(models, ax, fp_cost, fn_cost, timeframe):
    mil_formatter = FuncFormatter(millions)
    pct_formatter = FuncFormatter(to_percent)
    
    for it in models:
        model = it.best_iteration
        profits, max_profit = calculateProfit(model, it.y_test, fp_cost, fn_cost, timeframe)

        ax.plot(profits.pct_instance, profits.profit_change,
                     label = it.name +" Max: {}M Pct Inst.: {:0.1%}".format(
                         locale.currency(max_profit['diff']*10**-6,grouping=True),max_profit['pct_instance']))
        ax.yaxis.set_major_formatter(mil_formatter)
        ax.xaxis.set_major_formatter(pct_formatter)

    ax.set_title("Comparison of Profit Curves (Cost Reduction) on Test Data")
    ax.set_xlabel("Percentage of Test Instances")
    ax.set_ylabel("Expected Profit Improvement (Cost Reduction)")
    ax.legend()
    
def plotROC(models, ax):
    pct_formatter = FuncFormatter(to_percent)
    
    for it in models:
        model = it.best_iteration
        ax.plot(model.fpr, model.tpr, label = it.name +" (AUC = {:0.3})".format(model.auc))

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
    return np.mean(aucs), np.sqrt((np.var(aucs)))

def plotSampleSize(models, sampsize, nruns, axes):
    x_vals = np.log2(sampsize)
    #cmap = plt.cm.get_cmap('hsv', len(models) + 1)
    #cmap(c)
    
    for c, model in enumerate(models):
        model_mean = []
        model_std = []
        for samp in sampsize:
            mean_auc, std_auc = bootstrapAUC(model, samp, nruns)
            model_mean.append(mean_auc)
            model_std.append(std_auc)
            
        means = pd.Series(model_mean)
        stds= pd.Series(model_std)
    
        axes.plot(x_vals, means, label = model.name)
        #axes.plot(x_vals, means - 2 * stds, linestyle='--', label='_nolegend_')
        #axes.plot(x_vals, means + 2 * stds, linestyle='--', label='_nolegend_')
        
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
    
    plotProfitChange(models, axes[0], fp_cost, fn_cost, timeframe)
    plotProfitDiff(models, axes[1], fp_cost, fn_cost, timeframe)
    plotROC(models, axes[2])   
    plotSampleSize(models, sampsize, nruns, axes[3])
    