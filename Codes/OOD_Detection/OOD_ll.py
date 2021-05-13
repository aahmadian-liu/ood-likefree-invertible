# ** OOD detection using simple likelihood threshold

import pickle
import numpy as np
import matplotlib.pyplot as plt
import eval

# Loading loglikelihoods from file
def load_datall(resfile,dataind):
    # resfile: file path
    # dataind: indices of the samples whose LL should be loaded

    indic = pickle.load(open(resfile, 'rb'))
    features=indic['data']['LL'][dataind[0]:dataind[1]]

    features = np.array(features).transpose()

    return features

# Obtaining the operating curve of the OOD classifier, which here means the confusion matrix (true/false positive/negatives) at a number of different threshold values (positive=OOD).
# The thresholds are chosen uniformly over the range [min_score,max_score] where the min and max scores are the loglikelihood over all of the data (both in and out of distributions).
def Opcurve(x_in,x_out,ndivs):
    # x_in,x_out: log-likelihoods corresponding to in-distribution and OOD test samples
    # ndivs: number of threshold values to compute the measures at

    n_in=x_in.shape[0]
    n_out=x_out.shape[0]

    maxs = max(x_in.max(), x_out.max())
    mins = min(x_in.min(), x_out.min())

    thres = np.linspace(mins, maxs, ndivs)

    truepos=[]
    falsepos=[]
    trueneg=[]
    falseneg=[]

    for th in thres:

        c1 = np.sum(x_in < th)
        c2 = np.sum(x_out < th)

        truepos.append(float(c2))
        falsepos.append(float(c1))
        trueneg.append(float(n_in-c1))
        falseneg.append(float(n_out-c2))

    rdic={'TP':truepos,'FP':falsepos,'TN':trueneg,'FN':falseneg,'N_in(neg)':n_in,'N_out(pos)':n_out,'Threshs':thres}

    return rdic


# Running the main OOD detection algorithm, and obtaining the evaluation measures and plots on a set of dataset pairs
def RunOOD(indis_file,testout_files,ntrain,showplot=False):
    #indis_file: name of the file containing the statistics on in-distribution data
    #testout_files: the list of names of the files containing the statistics on OOD data
    #ntrain: number of the in-distribution samples used to train the classifier (others are used for test) 

    based='savedcomps/' #base file path
    cpoints=200 #number of points on the ROC and PR curves
    
    print("*OOD using Simple Likelihood Threshold")

    x_train = load_datall(based+indis_file+".sts", (0, ntrain))
    fdic = pickle.load(open(based+indis_file+".sts", 'rb'))
    trainame=fdic['info']['model_trained_data']
    print("training file:",indis_file)

    resdic = dict()
    resdic['trainfile'] = indis_file
    resdic['trainsize'] = x_train.shape
    resdic['method']='llthre'
    resdic['outfiles']=testout_files
    resdic['opcurves']=[]

    x_test_in = load_datall(based+indis_file+".sts", (ntrain, 2*ntrain))
    print("training shape:",x_train.shape,"test shape:",x_test_in.shape)

    for fout in testout_files:
        x_test_out = load_datall(based + fout+".sts", (0, ntrain))

        fdic = pickle.load(open(based + fout+".sts", 'rb'))
        print("\nOOD set:", fdic['info']['evaldata'])
        if not fdic['info']['model_trained_data']==trainame:
            print("Warning: mismatch with in-distribution")

        oc = Opcurve(x_test_in,x_test_out, cpoints)
        tit=trainame + " on " + fdic['info']['evaldata']
        auroc, roc = eval.ComputeROC(oc, showplot,tit)
        auprc,_=eval.ComputePRC(oc, showplot,tit)
        ftp=eval.ComputeFP95TP(roc)

        print('AUROC=',auroc)
        print("AUPRC=",auprc)
        print("FPR@95%TPR",ftp)

        resdic['opcurves'].append(oc)
        if showplot:
            plt.hist(x_test_in, label='in',bins=80)
            print(x_test_in.shape)
            plt.hist(x_test_out, label='out', bins=80,alpha=0.7)
            print(x_test_out.shape)
            plt.title('LL - ' + tit)
            plt.legend()
            plt.show()


    print("\n")

    return resdic


#example
if __name__ == "__main__":
    RunOOD('mnist_on_mnist_ires',['mnist_on_fashion_ires','mnist_on_uninoise_ires','mnist_on_flipv_ires','mnist_on_fliph_ires'],3000,False)
