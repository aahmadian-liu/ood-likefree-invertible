# **Some functions for evaluating OOD detection

import numpy as np
import matplotlib.pyplot as plt


# Sorting a set of (x,y) pairs based on the ascending values of x. This is necessary for getting the correct output from the numpy integration function.
# If more than pairs have the same x value, only the one with maximum y value is kept.
def sortxy(x, y):

    #finding the pairs with same x value, and reducing them to the one with highest y
    rdic=dict()
    for i in range(len(x)):
        rdic[x[i]]=max(y[i],rdic.get(x[i],0))

    #arranging the x and y values as a set of (x,y) tuples
    xt=list(rdic.keys())
    yt=[0]*len(xt)
    for t in range(0,len(xt)):
        yt[t]=rdic[xt[t]]

    xy = zip(xt, yt)

    # sorting based on x values, and returning as two lists of x and y
    xy = sorted(xy, key=lambda t: t[0])

    x_s, y_s = zip(*xy)

    return x_s, y_s

# Computing the ROC curve and the area under it, given an operating curve (i.e. true\false positives\negatives at different thresholds)
def ComputeROC(opcurve,show=False,ftitle=''):

    n_in = opcurve['N_in(neg)']
    n_out = opcurve['N_out(pos)']

    # true and false positive rates
    tprs = [v/n_out for v in opcurve['TP']]
    fprs = [v/n_in for v in opcurve['FP']]

    fprs,tprs = sortxy(x=fprs, y=tprs)

    # computing the area using trapezoidal numerical integration
    area=np.trapz(y=tprs,x= fprs)

    if show:
        plt.figure()
        plt.plot(fprs, tprs, '-o')
        plt.xlabel("FPR [P(error|in)]")
        plt.ylabel("TPR [P(correct|out)]")
        plt.title(ftitle + "\n auc=" + str(area)[0:4])
        plt.show()

    return area,(fprs,tprs)

# Computing the Precision-Recall curve and the area under it, given an operating curve (i.e. true\false positives\negatives at different thresholds)
def ComputePRC(opcurve,show=False,ftitle=''):

    n_out = opcurve['N_out(pos)']

    recal = [v/n_out for v in opcurve['TP']]
    prec=[]
    for i in range(len(opcurve['TP'])):
        if opcurve['TP'][i]+opcurve['FP'][i]>0:
            prec.append(opcurve['TP'][i] / (opcurve['TP'][i] + opcurve['FP'][i]))
        else:
            prec.append(1.0)

    recal, prec = sortxy(x=recal, y=prec)

    area = np.trapz(x=recal,y=prec)

    if show:
        plt.figure()
        plt.plot(recal, prec, '-o')
        plt.xlabel("Recal")
        plt.ylabel("Prec")
        plt.title(ftitle + "\n auc=" + str(area)[0:4])
        plt.show()

    return area,(prec,recal)

# Computing the false positive rate when the true positive rate is at 0.95 of its maximum, given a ROC curve
def ComputeFP95TP(rocurve):

    fprs,tprs=rocurve[0],rocurve[1]

    maxtp=max(tprs)

    for i in range(len(tprs)):
        if tprs[i] >= maxtp*0.95:
            return fprs[i]

#Plotting the type one and two errors on test data versus type one error on training data (significance level)
def PlotErrTypes(svm,x_train, x_in, x_out):

    n_test_in = float(x_in.shape[0])
    n_test_out = float(x_out.shape[0])
    n_train=float(x_train.shape[0])

    trainscores = svm.score_samples(x_train)
    outscores = svm.score_samples(x_out)
    inscores = svm.score_samples(x_in)

    alphas = np.linspace(0, 1, 100)
    thres = []

    type1errs = []
    type2errs=[]

    for a in alphas:
        th = np.quantile(trainscores, a)
        c1 = np.sum(inscores < th)
        c2=np.sum(outscores > th)

        type1errs.append(c1 / n_test_in)
        type2errs.append(c2 / n_train)
        thres.append(th)

    plt.figure()
    plt.plot(alphas, type1errs,label="Type I",linewidth=2)
    plt.plot(alphas, type2errs, label="Type II",linewidth=2)
    plt.xlabel("Significance (Training)")
    plt.ylabel("Error (Test)")
    plt.legend()
    plt.show()
