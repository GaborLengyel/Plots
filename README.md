# Plots
Plotting functions for (1) error bars, and (2) scatter plots in ptyhon using numpy, scipy, matplotlib.pyplot, and matplotlib.patches

Usage:

(1) import it - e.g. from PythonPlots import ErrorBarsForMeans

(2) call the function - e.g.  ErrorBarsForMeans(
                              data, 
                              SpreadOfX = 0.1,
                              YError = 'CI', 
                              plotsize = [10,20], 
                              axeslimit = [], 
                              axisLabels = [[],['Performance incongruent-congruent']], 
                              SameAxisLabel = True, 
                              SubplotTitles = ['Object effect', 'Cue validity effect'], 
                              SameSubplotTitles = False, 
                              plotTitle = 'Performance in visual search', 
                              ThresValue = [0,0], 
                              AxisTicks = [['RT','Error'],[]], 
                              SaveFigName = [], 
                              Outliers = [],
                              titleFont = {'fontname':'Arial', 'size':'28', 'color':'black',                      'weight':'normal','verticalalignment':'bottom'},
                              axisFont = {'fontname':'Arial', 'size':'24'},
                              ErrorBar = {'ErrLen':0.1, 'ErrWid1':6, 'ErrWid2':6, 'sizeMean':24, 'sizedots':10, 'ErrColor':['k','k','k','k'], 'DotsColor':'y','OutlierDotsColor':'r'})
                              
                              
                              
Inputs:
