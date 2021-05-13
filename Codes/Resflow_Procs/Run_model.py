#** computing ResFlow outputs on a dataset, which are latent codes, likelihood, Jacobian column sum, and gradient wrt prameters,
#     and saving them to a file, so that OOD detection statistics can be later obtained from them

import numpy as np
import torch
import pickle
np.random.seed(0)
torch.manual_seed(0)
import Modeleval as modeleval
from os.path import getmtime
import torch.autograd as autograd
import argparse
import time

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-modelfile',type=str,help="path of the pytorch trained model file") 
parser.add_argument('-trainedon',type=str,help="dataset name on which the model has been trained; one of {'mnist','fashion','svhn','cifar10','cifar100'}")
parser.add_argument('-teston',type=str,help="dataset name on which the model outputs/statistics are computed; same options as 'trainedon' in addtion to 'blank' and 'uniform'")
parser.add_argument('-flipmode',type=int,default=0,help="whether to flip the test images (0:disabled, 1:vertical, 2:horizontal)")
parser.add_argument('-notlikelihood',default=False,help="whether to skip computing the likelihood (and log-det term)",action='store_true')
parser.add_argument('-n',type=int,help="number of test samples (images) to process")
parser.add_argument('-outfile',type=str,help="file name to write the output")
args=parser.parse_args()

bsize=16 #batch size

evaldata_train=False # whether to evaluate on training partition

#assigning the argument values
modelpath=args.modelfile
model_trained_data=args.trainedon
evaldata=args.teston
flip=args.flipmode
outfile=args.outfile
nsamples=args.n
model_date='pretrained'
modeleval.args.val_batchsize = bsize
modeleval.args.batchsize = bsize
comll=not args.notlikelihood
print(comll)

maxb=nsamples//bsize #number of batches to process

#CUDA device should be set in modeleval module
print("device:",modeleval.device,"batchsize:",modeleval.args.batchsize,modeleval.args.val_batchsize)

# loading data and model

if evaldata=='uniform':
    dataset=modeleval.Uninoiseloader()
elif evaldata=='blank':
    dataset=modeleval.Blankloader()
else:
    dataset= modeleval.GetData(evaldata,evaldata_train)


model,_= modeleval.LoadResflow(modelpath,model_trained_data)

D=[] # data points (images-labels)
Z=[] # latent vectors
L=[] # log-likelihoods
En_Jac=[] # encoder Jacobian column sum
Pgra=[] # norms of the gradient of loglikelihood wrt parameters

npr=0 #total number of model parameters
# enable gradient computation wrt all model parameters
for p in model.parameters():
    if len(p.shape)>0:
        p.requires_grad_()
        p.grad = None
        npr+=p.numel()
print("model loaded."," #params:",npr)


# the main loop for computing model outputs, for each batch
iter=0

for (x,y) in dataset:

    # apply flipping if enabled
    if flip == 1:
        x = x.flip(2)
    if flip == 2:
        x = x.flip(3)
    # enable gradient computation wrt input images
    x.requires_grad_()

    # forward propagation
    z,ll = modeleval.Forward(x, model, comll)
    #   the third argument must be True if likelihood is needed
    #   GPU is used if available and enabled in 'modeleval'
    
    # computing the column sum of the encoder's Jacobian matrix, i.e vJ where v is a vector of ones,
    # or equivalently gradient of f(z)=z_1+...+z_m wrt to x
    jr=autograd.grad(z,x,grad_outputs=[torch.ones_like(z)],only_inputs=True,retain_graph=True,allow_unused=True)
    
    for j in range(0,bsize):

        # appending the obtained values for each sample in the batch to the corresponding arrays
        D.append((x[j,:].detach().numpy(), y[j].detach().numpy()))
        if comll:
            L.append(ll[j].cpu().detach().numpy()[0])
        Z.append(z[j,:].cpu().detach().numpy())
        En_Jac.append(jr[0][j,:].cpu().detach().numpy())

        if comll:
            gnorm=0
            # gradient of the loglikelihood wrt model parameters
            gp=autograd.grad(ll[j], model.parameters(), only_inputs=True,allow_unused=True,retain_graph=True)
            # computing and appending the norm of the gradient
            for g in gp:
                if not (g is None):
                    gnorm+=np.sum((g.cpu().detach().numpy())**2)
                
            gnorm=np.sqrt(gnorm/npr)
            Pgra.append(gnorm)

    # freeing the gradients to avoid memory leak
    x.grad = None
    for p in model.parameters():
        p.grad = None

    iter+=1
    print("ResFlow  " + model_trained_data + " on " + evaldata + " flip"+str(flip) + "\n", "#batch", iter, "/", maxb, " (", iter * bsize, "sams)")
    if iter==maxb:
        break


# saving the results and some information

Dict=dict()
Dict['Z']= Z
Dict['LL']= L
Dict['Encoder_Jac']= En_Jac
Dict['GNParam']= Pgra
Dict['Data']= D
Dict['ndata']= nsamples

InfoDict=dict()
if flip>0:
    evaldata=evaldata+"-flip"+str(flip)
InfoDict['evaldata']= evaldata
InfoDict['modelpath']=modelpath
InfoDict['model_createddate']=model_date
InfoDict['model_trained_data']=model_trained_data

Dict['info']=InfoDict
Dict['modeltype']='ResFlow'
Dict['filename']=outfile

pickle.dump(Dict,open(outfile,'wb'))
print("** Successfully accomplished.")
