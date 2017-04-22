def ErrorBarsForMeans(data,
                      SpreadOfX = 0.1,
                      YError = 'CI',
                      plotsize = [15,15],
                      axeslimit = [],
                      axisLabels = [],
                      SameAxisLabel = True,
                      SubplotTitles = [],
                      SameSubplotTitles = True,
                      plotTitle = [],
                      ThresValue = [],
                      AxisTicks = [],
                      SameAxisTicks = True,
                      SaveFigName = [],
                      Outliers = [],
                      titleFont = {'fontname':'Arial', 'size':'28', 'color':'black', 'weight':'normal','verticalalignment':'bottom'},
                      axisFont = {'fontname':'Arial', 'size':'24'},
                      ErrorBar = {'ErrLen':0.15, 'ErrWid1':6, 'ErrWid2':6, 'sizeMean':24, 'sizedots':10, 'ErrColor':'k', 'DotsColor':'y','OutlierDotsColor':'r'},
                      FigureLayout = [1,1,1,0.95]):

    # imports
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats as sc

    # parameters for the figure:
    ColSubplots = len(data)
    RowSubplots = len(data[0])
    locx = range(1,len(data[0][0])+1)
    NSubPlots = ColSubplots*RowSubplots

    # Figure
    fig, ax = plt.subplots(RowSubplots,ColSubplots, figsize=(plotsize[0], plotsize[1]), squeeze=False)
    
    ax = ax.ravel()

    SubPlot = -1
    for Row in range(RowSubplots):
        for Col in range(ColSubplots):

            SubPlot = SubPlot+1

            for Bars in range(len(data[Col][Row])):

                # data:
                Ys = data[Col][Row][Bars]
                Xs = np.random.normal(loc=locx[Bars], scale=SpreadOfX, size=len(Ys))
                # 95% Conf.Intervals
                if YError == 'CI':
                    LowerError = sc.t.interval(0.95, len(Ys)-1, loc=np.mean(Ys), scale=sc.sem(Ys))[0]
                    UpperError = sc.t.interval(0.95, len(Ys)-1, loc=np.mean(Ys), scale=sc.sem(Ys))[1]
                # standard deviation
                elif YError == 'STD':
                    LowerError = np.mean(Ys)-np.std(Ys)
                    UpperError = np.mean(Ys)+np.std(Ys)
                # SEM
                elif YError == 'SEM':
                    LowerError = np.mean(Ys)-sc.sem(Ys)
                    UpperError = np.mean(Ys)+sc.sem(Ys)
                    
                # Error bar:
                ax[SubPlot].plot([locx[Bars],locx[Bars]],[LowerError,UpperError], ErrorBar['ErrColor'][Bars]+'-', linewidth=ErrorBar['ErrWid1'])
                ax[SubPlot].plot([locx[Bars]-ErrorBar['ErrLen'],locx[Bars]+ErrorBar['ErrLen']],[UpperError,UpperError], ErrorBar['ErrColor'][Bars]+'-', linewidth=ErrorBar['ErrWid2'])
                ax[SubPlot].plot([locx[Bars]-ErrorBar['ErrLen'],locx[Bars]+ErrorBar['ErrLen']],[LowerError,LowerError], ErrorBar['ErrColor'][Bars]+'-', linewidth=ErrorBar['ErrWid2'])
                ax[SubPlot].plot(locx[Bars],np.mean(Ys),"ko", markersize=ErrorBar['sizeMean'])
                # Points
                ax[SubPlot].plot(Xs, Ys, ErrorBar['DotsColor']+'o', markersize=ErrorBar['sizedots'])
                # Outliers
                if Outliers:
                    ax[SubPlot].plot(Xs[Outliers], Ys[Outliers], ErrorBar['OutlierDotsColor']+'o', markersize=ErrorBar['sizedots'])

            # showing the threshold value
            if ThresValue:
                ax[SubPlot].plot([locx[0]-1,locx[-1]+1], [ThresValue[SubPlot],ThresValue[SubPlot]], "k--", linewidth=ErrorBar['ErrWid2']/2)


            # setting plot parameters
            # Axis:
            for label in (ax[SubPlot].get_xticklabels() + ax[SubPlot].get_yticklabels()):
                label.set_fontname(axisFont['fontname'])
                label.set_fontsize(axisFont['size'])
            ax[SubPlot].spines['right'].set_visible(False)
            ax[SubPlot].spines['top'].set_visible(False)
            ax[SubPlot].xaxis.set_ticks_position('bottom')
            ax[SubPlot].yaxis.set_ticks_position('left')

            # Axis limits and ticks:
            if axeslimit:
                ax[SubPlot].yaxis.set_ticks(np.arange(axeslimit[0][SubPlot], axeslimit[1][SubPlot], axeslimit[2][SubPlot]))
                ax[SubPlot].set_ylim([axeslimit[0][SubPlot], axeslimit[1][SubPlot]])
            ax[SubPlot].xaxis.set_ticks(locx)
            ax[SubPlot].set_xlim([locx[0]-1, locx[-1]+1])
            
            # Axis Tick labels:
            if SameAxisTicks:
                if AxisTicks[0]:
                    ax[SubPlot].xaxis.set_ticklabels(AxisTicks[0])
                if AxisTicks[1]:
                    ax[SubPlot].yaxis.set_ticklabels(AxisTicks[1])
            else:
                if AxisTicks[0]:
                    ax[SubPlot].xaxis.set_ticklabels(AxisTicks[0][SubPlot])
                if AxisTicks[1]:
                    ax[SubPlot].yaxis.set_ticklabels(AxisTicks[1][SubPlot])

            # Axis lables:
            # X:
            if SameAxisLabel:
                if Row == range(RowSubplots)[-1]:
                    if len(axisLabels[0])>1:
                        ax[SubPlot].set_xlabel(axisLabels[0][Col], **axisFont)
                    elif len(axisLabels[0])==1:
                        ax[SubPlot].set_xlabel(axisLabels[0][0], **axisFont)
            else:
                ax[SubPlot].set_xlabel(axisLabels[0][SubPlot], **axisFont)
            # Y:
            if SameAxisLabel:
                if SubPlot%(Col+1)==0 or SubPlot==1:
                    if len(axisLabels[1])>1:
                        ax[SubPlot].set_ylabel(axisLabels[1][Row], **axisFont)
                    elif len(axisLabels[1])==1:
                        ax[SubPlot].set_ylabel(axisLabels[1][0], **axisFont)
            else:
                ax[SubPlot].set_ylabel(axisLabels[1][SubPlot], **axisFont)

            # Legend: TODO: finish this
            ax[SubPlot].legend(loc='lower right')

            # Subplot titles:
            if SubplotTitles:
                if SameSubplotTitles:
                    if Col in range(ColSubplots):
                        ax[SubPlot].set_title(SubplotTitles[Col], **axisFont)
                else:
                    ax[SubPlot].set_title(SubplotTitles[SubPlot], **axisFont)


    # Settings for the whole plot
    plt.suptitle(plotTitle,**titleFont)
    plt.tight_layout(pad=FigureLayout[0], w_pad=FigureLayout[1], h_pad=FigureLayout[2])
    fig.subplots_adjust(top=FigureLayout[3])

    if SaveFigName:
        plt.savefig(SaveFigName)

    plt.show()

#############################################################################################################################
    def ScatterForCorrelation(data,
                        SpreadOfX = 0.1,
                        YError = 'CI',
                        NoStd = 1,
                        plotsize = [15,15],
                        axeslimitX = [],
                        axeslimitY = [],
                        axisLabels = [],
                        SameAxisLabel = True,
                        ConditionLabels = [],
                        SameConditionLabels = True,
                        SubplotTitles = [],
                        SameSubplotTitles = True,
                        plotTitle = [],
                        ThresValue = [],
                        SameThresValue = True,
                        RegressionLine = True,
                        AxisTicks = [],
                        SameAxisTicks = True,
                        SaveFigName = [],
                        Outliers = [],
                        SameErrorEllipseColor = True,
                        ErrorEllipseColor = [],
                        ErrorBar = {'ErrDist':[], 'ErrSize':1, 'ErrWid':6, 'sizedots':10, 'ErrColor':[], 'DotsColor':[],'OutlierDotsColor':'r'},
                        titleFont = {'fontname':'Arial', 'size':'28', 'color':'black', 'weight':'normal','verticalalignment':'bottom'},
                        axisFont = {'fontname':'Arial', 'size':'24'},
                        LegendPos = {'LegendPosition':'upper left'},
                        FigureLayout = [1,1,1,0.95]):

    # imports
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats as sc
    from matplotlib.patches import Ellipse

    # parameters for the figure:
    ColSubplots = len(data)
    RowSubplots = len(data[0])
    NScatter = len(data[0][0])

    # computing the position and length of the error bars
    ErrPosXY = np.zeros((ColSubplots,RowSubplots,NScatter,2))
    ErrLenXY = np.zeros((ColSubplots,RowSubplots,NScatter,2))
    for Row in range(RowSubplots):
        for Col in range(ColSubplots):
            SortedMeanXY = np.zeros((NScatter))
            AllXY = np.empty((2,0))
            for Scatter in range(len(data[Col][Row])):

                Ys = data[Col][Row][Scatter][1]
                Xs = data[Col][Row][Scatter][0]
                SortedMeanXY[Scatter] = np.array([np.mean(Ys)])
                AllXY = np.hstack((AllXY,[Xs,Ys]))

            SortedMeanXY = np.argsort(SortedMeanXY, axis = 0)
            for i in range(SortedMeanXY.shape[0]-1,-1,-1):
                ErrPosXY[Col,Row,SortedMeanXY[i],1] = np.amin(AllXY[1,:])-(np.ptp(AllXY[1,:]*ErrorBar['ErrDist'][i]))
                ErrPosXY[Col,Row,SortedMeanXY[i],0] = np.amin(AllXY[0,:])-(np.ptp(AllXY[0,:]*ErrorBar['ErrDist'][i]))
                ErrLenXY[Col,Row,SortedMeanXY[i],1] = (np.ptp(AllXY[1,:]*ErrorBar['ErrSize']*0.1)) # TODO make it less stupit
                ErrLenXY[Col,Row,SortedMeanXY[i],0] = (np.ptp(AllXY[0,:]*ErrorBar['ErrSize']*0.1))

    # for the regression line
    if RegressionLine and len(RegressionLine)==1:
        RegressionLine = np.repeat(RegressionLine,ColSubplots*RowSubplots)

    # Figure
    fig, ax = plt.subplots(RowSubplots,ColSubplots, figsize=(plotsize[0], plotsize[1]), squeeze=False)
    
    ax = ax.ravel()

    SubPlot = -1
    Scatteridx = -1
    for Row in range(RowSubplots):
        for Col in range(ColSubplots):

            SubPlot = SubPlot+1
            for Scatter in range(len(data[Col][Row])):

                Scatteridx = Scatteridx+1
                # data:
                Ys = data[Col][Row][Scatter][1]
                Xs = data[Col][Row][Scatter][0]
                BarsY = ErrPosXY[Col,Row,Scatter,0]
                BarsX = ErrPosXY[Col,Row,Scatter,1]
                ErrLenY = ErrLenXY[Col,Row,Scatter,0]
                ErrLenX = ErrLenXY[Col,Row,Scatter,1]
                # Error intervals For Ys
                # 95% Conf.Intervals
                if YError == 'CI':
                    LowerErrorY = sc.t.interval(0.95, len(Ys)-1, loc=np.mean(Ys), scale=sc.sem(Ys))[0]
                    UpperErrorY = sc.t.interval(0.95, len(Ys)-1, loc=np.mean(Ys), scale=sc.sem(Ys))[1]
                # standard deviation
                elif YError == 'STD':
                    LowerErrorY = np.mean(Ys)-np.std(Ys)
                    UpperErrorY = np.mean(Ys)+np.std(Ys)
                # SEM
                elif YError == 'SEM':
                    LowerErrorY = np.mean(Ys)-sc.sem(Ys)
                    UpperErrorY = np.mean(Ys)+sc.sem(Ys)
                # Error intervals For Xs
                # 95% Conf.Intervals
                if YError == 'CI':
                    LowerErrorX = sc.t.interval(0.95, len(Xs)-1, loc=np.mean(Xs), scale=sc.sem(Xs))[0]
                    UpperErrorX = sc.t.interval(0.95, len(Xs)-1, loc=np.mean(Xs), scale=sc.sem(Xs))[1]
                # standard deviation
                elif YError == 'STD':
                    LowerErrorX = np.mean(Xs)-np.std(Xs)
                    UpperErrorX = np.mean(Xs)+np.std(Xs)
                # SEM
                elif YError == 'SEM':
                    LowerErrorX = np.mean(Xs)-sc.sem(Xs)
                    UpperErrorX = np.mean(Xs)+sc.sem(Xs)

                # Error ellipse
                points=np.stack((Xs, Ys))
                cov = np.cov(points)
                # central point of the error ellipse
                #Thresholds
                pos=[np.mean(Xs),np.mean(Ys)]
                # for the angle we need the eigenvectors of the covariance matrix
                w,v=np.linalg.eig(cov)
                # We pick the largest eigen value
                order = w.argsort()[::-1]
                w=w[order]
                v=v[:,order]
                # we compute the angle towards the eigen vector with the largest eigen value
                theta = np.degrees(np.arctan(v[1,0]/v[0,0]))
                thetar = np.arctan(v[1,0]/v[0,0])
                # Compute the width and height of the ellipse based on the eigen values (ie the length of the vectors)
                width, height = 2 * NoStd * np.sqrt(w)
                # making the ellipse
                ellip = Ellipse(xy=pos, width=width, height=height, angle=theta)
                ellip.set_alpha(0.1)                
                if SameErrorEllipseColor:
                    ellip.set_facecolor(ErrorEllipseColor[Scatter])
                else:
                    ellip.set_facecolor(ErrorEllipseColor[SubPlot][Scatter])
                    
                # computing regression lines
                if RegressionLine[SubPlot]:
                    xT=np.stack((np.ones(len(Xs)), Xs), axis=-1)
                    reg=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xT),xT)),np.transpose(xT)),Ys)


                # Scatter plot
                if ConditionLabels:
                    if SameConditionLabels:
                        Label = ConditionLabels[Scatter]
                    else:
                        Label = ConditionLabels[SubPlot][Scatter]
                else:
                    Label = []
                ax[SubPlot].scatter(Xs, Ys, s=ErrorBar['sizedots'], c=ErrorBar['DotsColor'][Scatter], marker="o", label=Label)
                # error ellipse
                ax[SubPlot].add_artist(ellip)
                # Error bar for Ys:
                ax[SubPlot].plot([BarsY,BarsY],[LowerErrorY,UpperErrorY], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot([BarsY-ErrLenY,BarsY+ErrLenY],[UpperErrorY,UpperErrorY], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot([BarsY-ErrLenY,BarsY+ErrLenY],[LowerErrorY,LowerErrorY], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot(BarsY,np.mean(Ys),ErrorBar['ErrColor'][Scatter]+'o', markersize=ErrorBar['sizeMean'])
                # Error bar for Xs:
                ax[SubPlot].plot([LowerErrorX,UpperErrorX],[BarsX,BarsX], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot([UpperErrorX,UpperErrorX],[BarsX-ErrLenX,BarsX+ErrLenX], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot([LowerErrorX,LowerErrorX],[BarsX-ErrLenX,BarsX+ErrLenX], ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                ax[SubPlot].plot(np.mean(Xs),BarsX,ErrorBar['ErrColor'][Scatter]+'o', markersize=ErrorBar['sizeMean'])
                # regression line
                if RegressionLine[SubPlot]:
                    ax[SubPlot].plot([min(Xs)-np.std(Xs),max(Xs)+np.std(Xs)],[reg[1]*(min(Xs)-np.std(Xs))+reg[0],reg[1]*(max(Xs)+np.std(Xs))+reg[0]],ErrorBar['ErrColor'][Scatter]+'-', linewidth=ErrorBar['ErrWid'])
                # zero effect lines
                if ThresValue:
                    if SameThresValue:
                        ax[SubPlot].plot([min(Xs), max(Xs)], [ThresValue[1],ThresValue[1]], "k--", linewidth=ErrorBar['ErrWid']/2)
                        ax[SubPlot].plot([ThresValue[0],ThresValue[0]], [min(Ys), max(Ys)], "k--", linewidth=ErrorBar['ErrWid']/2)
                    else:
                        ax[SubPlot].plot([min(Xs), max(Xs)], [ThresValue[SubPlot][1],ThresValue[SubPlot][1]], "k--", linewidth=ErrorBar['ErrWid']/2)
                        ax[SubPlot].plot([ThresValue[SubPlot][0],ThresValue[SubPlot][0]], [min(Ys), max(Ys)], "k--", linewidth=ErrorBar['ErrWid']/2)
                # Outliers
                if Outliers:
                    ax[SubPlot].plot(Xs[Outliers[SubPlot][Scatter]], Ys[Outliers[SubPlot][Scatter]], ErrorBar['OutlierDotsColor'][Scatter]+'o', markersize=ErrorBar['sizedots'])

            # setting plot parameters
            # Axis:
            for label in (ax[SubPlot].get_xticklabels() + ax[SubPlot].get_yticklabels()):
                label.set_fontname(axisFont['fontname'])
                label.set_fontsize(axisFont['size'])
            ax[SubPlot].spines['right'].set_visible(False)
            ax[SubPlot].spines['top'].set_visible(False)
            ax[SubPlot].xaxis.set_ticks_position('bottom')
            ax[SubPlot].yaxis.set_ticks_position('left')

            # Axis limits and ticks:
            if axeslimitX:
                ax[SubPlot].xaxis.set_ticks(np.arange(axeslimitX[0][SubPlot], axeslimitX[1][SubPlot], axeslimitX[2][SubPlot]))
                ax[SubPlot].set_xlim([axeslimitX[0][SubPlot]+axeslimitX[0][SubPlot]*0.01, axeslimitX[1][SubPlot]+axeslimitX[1][SubPlot]*0.01])
            if axeslimitY:
                ax[SubPlot].yaxis.set_ticks(np.arange(axeslimitY[0][SubPlot], axeslimitY[1][SubPlot], axeslimitY[2][SubPlot]))
                ax[SubPlot].set_ylim([axeslimitY[0][SubPlot]+axeslimitY[0][SubPlot]*0.01, axeslimitY[1][SubPlot]+axeslimitY[1][SubPlot]*0.01])
            
            # Axis Tick labels:
            if SameAxisTicks:
                if AxisTicks[0]:
                    ax[SubPlot].xaxis.set_ticklabels(AxisTicks[0])
                if AxisTicks[1]:
                    ax[SubPlot].yaxis.set_ticklabels(AxisTicks[1])
            else:
                if AxisTicks[0]:
                    ax[SubPlot].xaxis.set_ticklabels(AxisTicks[0][SubPlot])
                if AxisTicks[1]:
                    ax[SubPlot].yaxis.set_ticklabels(AxisTicks[1][SubPlot])

            # Axis lables:
            # X:
            if SameAxisLabel:
                if Row == range(RowSubplots)[-1]:
                    if len(axisLabels[0])>1:
                        ax[SubPlot].set_xlabel(axisLabels[0][Col], **axisFont)
                    elif len(axisLabels[0])==1:
                        ax[SubPlot].set_xlabel(axisLabels[0][0], **axisFont)
            else:
                ax[SubPlot].set_xlabel(axisLabels[0][SubPlot], **axisFont)
            # Y:
            if SameAxisLabel:
                if SubPlot%(Col+1)==0 or SubPlot==1:
                    if len(axisLabels[1])>1:
                        ax[SubPlot].set_ylabel(axisLabels[1][Row], **axisFont)
                    elif len(axisLabels[1])==1:
                        ax[SubPlot].set_ylabel(axisLabels[1][0], **axisFont)
            else:
                ax[SubPlot].set_ylabel(axisLabels[1][SubPlot], **axisFont)

            # Legend: TODO: finish this
            if ConditionLabels:
                ax[SubPlot].legend(loc=LegendPos['LegendPosition'], fontsize=axisFont['size'])

            # Subplot titles:
            if SubplotTitles:
                if SameSubplotTitles:
                    if Col in range(ColSubplots):
                        ax[SubPlot].set_title(SubplotTitles[Col], **axisFont)
                else:
                    ax[SubPlot].set_title(SubplotTitles[SubPlot], **axisFont)


    # Settings for the whole plot
    plt.suptitle(plotTitle,**titleFont)
    plt.tight_layout(pad=FigureLayout[0], w_pad=FigureLayout[1], h_pad=FigureLayout[2])
    fig.subplots_adjust(top=FigureLayout[3])

    if SaveFigName:
        plt.savefig(SaveFigName)

    plt.show()