# ** OOD detection using One Class SVM and precomputed statistics

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import eval



def printsource(indic):
    print("---")
    print("---")
    print(str.join("",indic['source_main']))

    print("---")

# loading computed statistics from file
def load_data(resfile,fnames,dataind):
    #resfile: file path
    #fnames: name of the statistics (features) that should be loaded
    #dataind: indices of the samples whose statistics should be loaded

    indic = pickle.load(open(resfile, 'rb'))

    #printsource(indic)
    #print(indic['info'])

    features = []
    for f in fnames:
        x=indic['data'][f][dataind[0]:dataind[1]]

        features.append(x)

    features = np.array(features).transpose()

    return features

# training an OSVM model, and returning both the model and scores assigned to the data by model
def Svmtrain(data):
    svm=OneClassSVM()
    svm.fit(data)
    scores=svm.score_samples(data)
    return svm,scores

# Obtaining the operating curve of the OOD classifier, which here means the confusion matrix (true/false positive/negatives) at a number of different threshold values (positive=OOD).
# The thresholds are chosen uniformly over the range [min_score,max_score] where the min and max scores are the OSVM scores over all of the data (both in and out of distributions).
def Opcurve(svm,x_in,x_out,ndivs,hist_plot_title):
    # svm: trained OSVM model
    # x_in, x_out: statistics corresponding to in-distribution and OOD test samples
    # ndivs: number of threshold values to compute the measures at

    n_in=float(x_in.shape[0])
    n_out=float(x_out.shape[0])

    inscores=svm.score_samples(x_in)
    outscores=svm.score_samples(x_out)
    maxs=max(inscores.max(),outscores.max())
    mins=min(inscores.min(), outscores.min())

    thres=np.linspace(mins,maxs,ndivs)

    truepos=[]
    falsepos=[]
    trueneg=[]
    falseneg=[]

    for th in thres:

        c1 = np.sum(inscores<th)
        c2 = np.sum(outscores<th)

        truepos.append(float(c2))
        falsepos.append(float(c1))
        trueneg.append(float(n_in-c1))
        falseneg.append(float(n_out-c2))

    if hist_plot_title is not None:
        plt.hist(inscores,label='in')
        plt.hist(outscores,label='out',alpha=0.7)
        plt.title('SVM score - ' + hist_plot_title)
        plt.legend()
        plt.show()

    rdic={'TP':truepos,'FP':falsepos,'TN':trueneg,'FN':falseneg,'N_in(neg)':n_in,'N_out(pos)':n_out,'Threshs':thres}

    return rdic


# Running the main OOD detection algorithm, and obtaining the evaluation measures and plots on a set of dataset pairs
def RunOOD(indis_file,testout_files,features,ntrain,showplot=False):
    #indis_file: name of the file containing the statistics on in-distribution data
    #testout_files: list of names of the files containing the statistics on OOD data
    #features: list of the names of the statistics which are used as input features
    #ntrain: number of the in-distribution samples used to train the classifier (others are used for test)
    
    
    based='savedcomps/' #base file path
    cpoints=200 #number of points on the ROC and PR curves
    print("*OOD using OSVM-S")
    print("using features:",features)

    x_train = load_data(based+indis_file+".sts", features, (0, ntrain))
    fdic = pickle.load(open(based+indis_file+".sts", 'rb'))
    trainame=fdic['info']['model_trained_data']
    print("training file:",indis_file)

    resdic = dict() #some meta information
    resdic['trainfile'] = indis_file
    resdic['trainsize'] = x_train.shape
    resdic['features'] = features
    resdic['method']='svm-s'
    resdic['outfiles']=testout_files
    resdic['opcurves']=[]

    x_test_in = load_data(based+indis_file+".sts", features, (ntrain, 2*ntrain))
    print("training shape:",x_train.shape,"test shape:",x_test_in.shape)

    # normalizing and PCA
    print("preprocessing...")
    prep1 = StandardScaler()
    prep1.fit(x_train)
    X = prep1.transform(x_train)
    prep2 = PCA()
    prep2.fit(X)
    X = prep2.transform(X)
    X_in = prep2.transform(prep1.transform(x_test_in))
    print("training the svm...")
    svm, inscores = Svmtrain(X)

    for fout in testout_files:
        x_test_out = load_data(based + fout+".sts", features, (0, ntrain))
        X_out = prep2.transform(prep1.transform(x_test_out))
        fdic = pickle.load(open(based + fout+".sts", 'rb'))
        print("\nOOD set:", fdic['info']['evaldata'])
        if not fdic['info']['model_trained_data']==trainame:
            print("Warning: mismatch with in-distribution")

        title=trainame + " on " + fdic['info']['evaldata'] if showplot else None
        oc = Opcurve(svm, X_in, X_out, cpoints,title)
        auroc, roc = eval.ComputeROC(oc, showplot,title)
        auprc,_=eval.ComputePRC(oc, showplot,title)
        ftp=eval.ComputeFP95TP(roc)

        print('AUROC=',auroc)
        print("AUPRC=",auprc)
        print("FPR@95%TPR",ftp)

        resdic['opcurves'].append(oc)

    #eval.PlotTypeI(svm.score_samples(X),svm.score_samples(X_in),trainame)


    print("\n")

    return resdic


#example
if __name__=="__main__":

    features =['LPZ', 'compxs']
    RunOOD('mnist_on_mnist_ires',['mnist_on_fashion_ires','mnist_on_uninoise_ires','mnist_on_flipv_ires','mnist_on_fliph_ires'],features,3000,False)
