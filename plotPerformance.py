import pandas as pd
import numpy as np

import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from betweenCompare import betweenCompare

import locale
locale.setlocale( locale.LC_ALL, '' )

def expectedValue(model, y_test, thresholds, fp_cost, fn_cost):
    # fp_cost and fn_cost should be the change in revenue associated with 1000 ad requests (i.e. rCPM change)
    # output of expected value is the expected rCPM 
    
    class_preds = list(map(lambda x, y: 1 if x > y else 0, model.predictions[:,1], thresholds))
    
    conf_mat = confusion_matrix(y_test, class_preds)/len(model.predictions)
    cost_mat = np.array([0, fp_cost,
                         fn_cost, 0]).reshape(2,2)
    
    ev = conf_mat[0,1]*cost_mat[0,1] + conf_mat[1,0]*cost_mat[1,0]
    
    return ev

def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x*1e-6)

def to_percent(x, pos):
    'The two args are the value and tick position'
    return '%.0f%%' % (x*100)

def dataProfit(timeframe, rcpm, pct):
    if timeframe=='yearly':
        profit = list(map(lambda x: x/1000*12*450*10**9, rcpm))
    elif timeframe=='monthly':
        profit = list(map(lambda x: x/1000*450*10**9, rcpm))
    else:
        raise ValueError('timeframe must be "yearly" or "monthly"')

    profitData = pd.DataFrame({'pct_instance': pct, 'profit': profit, 'rcpm_change': rcpm})
    profitData = profitData.sort_values('pct_instance')
    profitData['profit_change'] = profitData.profit.transform(lambda x: (x/x[0]-1)*-1)
    profitData['profit_diff'] = profitData.profit.transform(lambda x: (x-x[0]))
    
    return profitData

def maxProfit(profitData):
    profits = {}
    
    max_idx = profitData.profit_change.idxmax()
    profits['max_profit_change'] = profitData.profit_change.iloc[max_idx]
    profits['max_profit_pct_instance'] = profitData.pct_instance.iloc[max_idx]
    profits['max_profit'] = profitData.profit.iloc[max_idx]
    profits['max_profit_diff'] = profitData.profit_diff.iloc[max_idx]
    
    return profits  

def plots(axes, profitData, profits, model, it, mil_format, pct_format):
    
    axes[0].plot(profitData.pct_instance, profitData.profit_change,
                 label = it.name +" Max: {:0.1%} Pct Inst.: {:0.1%}".format(
                 profits['max_profit_change'],profits['max_profit_pct_instance']))
    axes[0].yaxis.set_major_formatter(pct_format)
    axes[0].xaxis.set_major_formatter(pct_format)

    axes[1].plot(profitData.pct_instance, profitData.profit_diff,
                 label = it.name +" Max: {}M Pct Inst.: {:0.1%}".format(
                 locale.currency(profits['max_profit_diff']*10**-6,grouping=True),profits['max_profit_pct_instance']))
    axes[1].yaxis.set_major_formatter(mil_format)
    axes[1].xaxis.set_major_formatter(pct_format)
    
    
    axes[2].plot(model.fpr, model.tpr, label = it.name + " (AUC = {:0.3})".format(model.auc))
    axes[2].yaxis.set_major_formatter(pct_format)
    axes[2].xaxis.set_major_formatter(pct_format)
    
    
def plotPerformance(models, fp_cost=-0.03, fn_cost=-0.06, timeframe='yearly'):
    # fp_cost based on Mopub rcpm
    # fn_cost = 2x fp_cost - Assume buyers act broadly because of a single bad actor
    # Twitter ad exchange request volume estimate:  
    #   https://media.mopub.com/media/filer_public/22/b5/22b58fbf-b077-4c2c-ae41-d53d06d23dd9/mopub_global_mobile_programmatic_trends_report_-_q2_2016.pdf
    #Profit Curves
    mil_format = FuncFormatter(millions)
    pct_format = FuncFormatter(to_percent)
    #Profit Curves Plot
    fig = plt.figure(figsize=(14,14))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    axes = [plt.subplot(gs[0, :2], ), plt.subplot(gs[0, 2:]), plt.subplot(gs[1, 1:3])]
    
    for it in models:
        model = it.best_iteration
        threshholds = model.threshholds
            
        rcpm_change = []
        pct_instance = []
            
        for thresh in threshholds:
            rcpm_change += [expectedValue(model, it.y_test, [thresh]*len(it.y_test), fp_cost, fn_cost)]
            pct_instance_thresh = np.sum(model.predictions > thresh)/len(it.y_test)    
            pct_instance += [pct_instance_thresh]
            
            profitData = dataProfit(timeframe, rcpm_change, pct_instance)
            profits = maxProfit(profitData)
            
        plots(axes, profitData, profits, model, it, mil_format, pct_format)

    axes[0].set_title("Comparison of Profit Curves (Cost Reduction) on Test Data")
    axes[0].set_xlabel("Percentage of Test Instances")
    axes[0].set_ylabel("Expected Profit Improvement (Cost Reduction)")
    axes[0].legend()
    
    axes[1].set_title("Comparison of Profit Curves (Cost Reduction) on Test Data")
    axes[1].set_xlabel("Percentage of Test Instances")
    axes[1].set_ylabel("Expected Profit Improvement (Cost Reduction)")
    axes[1].legend()
        
    axes[2].set_title("Comparison of ROC Curves on Test Data")
    axes[2].set_xlabel("fpr")
    axes[2].set_ylabel("tpr")
    axes[2].yaxis.set_major_formatter(pct_format)
    axes[2].xaxis.set_major_formatter(pct_format)
    axes[2].legend()
    