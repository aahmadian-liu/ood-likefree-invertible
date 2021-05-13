#** computing iResNet outputs on a dataset, which are latent codes, likelihood, Jacobian column sum, and gradient wrt prameters,
#   and saving them to a file, so that OOD detection statistics can be later obtained from them

import numpy as np
import torch
import pickle
np.random.seed(0)
torch.manual_seed(0)
import Dataload as dataload
from os.path import getmtime
import torch.autograd as autograd
import argparse
import time



# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-modelfile',type=str,help="path of the pytorch trained model file")
parser.add_argument('-trainedon',type=str,help="dataset name on which the model has been trained; see the 'defined datasets' in code")
parser.add_argument('-teston',type=str,help="dataset name on which the model outputs/statistics are computed; one of the 'defined datasets' in addition to 'uniform' and 'blank'")
parser.add_argument('-flipmode',type=int,default=0,help="whether to flip the test images (0:disabled, 1:vertical, 2:horizontal)")
parser.add_argument('-n',type=int,help="number of test samples (images) to process")
parser.add_argument('-notlikelihood',default=False,help="whether to skip computing the likelihood (and log-det term)",action='store_true')
parser.add_argument('-outfile',type=str,help="file name to write the output")
args=parser.parse_args()

# defined datasets
datasets={'cifar10_test':dataload.cifar10_test, 'svhn_test':dataload.svhn_test,'mnist_test':dataload.mnist_test,
          'fashion_test':dataload.fashion_test,'cifar10_train':dataload.cifar10_train,'svhn_train':dataload.svhn_train,'cifar100_test':dataload.cifar100_test}

bsize=32 # batch size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #CUDA device

modelpath=args.modelfile
model_trained_data=args.trainedon
evaldata=args.teston
flip=args.flipmode
outfile=args.outfile
nsamples=args.n
model_date=getmtime(modelpath)
comll=not args.notlikelihood

maxb=nsamples//bsize #number of batches to process
dataload.batchsize=bsize

print("device: ",device,"  batchsize:",dataload.batchsize)

# loading data and model

if evaldata=='uniform':
    dataset=dataload.Uninoiseloader()
elif evaldata=='blank':
    dataset=dataload.Blankloader()
elif evaldata=='fashion_ranblock':
    dataset=dataload.Ranblockloader(datasets['fashion_test'])
elif evaldata=='mnist_ranblock':
    dataset=dataload.Ranblockloader(datasets['mnist_test'])
else:
    dataset = dataload.Loader(datasets[evaldata])

model= torch.load(modelpath,map_location=device)['model']
model.module.numSeriesTerms=36 # number of the series terms for likelihood approximation in iResnet
                               # note: this must be also set in 'matrix_utils' module
model.module.eval()

D=[] # data points (images-labels)
Z=[] # latent vectors
L=[] # log-likelihoods
En_Jac=[] # encoder Jacobian column sum
Pgra=[] # norms of the gradient of loglikelihood wrt parameters

npr=0 #total number of model parameters
# enable gradient computation wrt all model parameters
for p in model.parameters():
    p.requires_grad_()
    p.grad = None
    npr+=p.numel()
print("model loaded."," #params:",npr)

# the main loop for computing model outputs, for each batch
iter=0

for (x,y) in dataset:

    #apply flipping if enabled
    if flip == 1:
        x = x.flip(2)
    if flip == 2:
        x = x.flip(3)

    #move tensor to GPU (if available) and enable gradient computation wrt input images
    x=x.to(device)
    x.requires_grad_()

    #forward propagation
    zl,lz,ld = model.module(x,not comll) # set to True if log-determinant term (likelihood) is not needed
    
    #zl, lz, ld = model(x)
    ll=lz+ld #likelihood= prior term + log-det term

    #the latent code returned by iResnet is arranged in a multiscale structure, but we need to flatten it
    nd = [zl[k].numel() // bsize for k in range(len(zl))] #number of dimesnions at each level (scale) of the output (note 'zl' corresponds to the entire batch)
    z = torch.zeros(bsize, sum(nd), device=device) #an array for storing the latent vectors of the batch, one sample per row
    # concatenating all dimensions of latent codes
    for j in range(0, bsize):

        r=0
        for k in range(len(zl)):
            z[j,r:r+nd[k]]=zl[k][j,:].flatten()
            r+=nd[k]


    # computing the column sum of the encoder's Jacobian matrix, i.e vJ where v is a vector of ones,
    # or equivalently gradient of f(z)=z_1+...+z_m wrt to x
    jr = autograd.grad(z, x, grad_outputs=[torch.ones_like(z)], only_inputs=True, retain_graph=True)

    for j in range(0, bsize):

        # appending the obtained values for each sample in the batch to the corresponding arrays
        D.append((x[j,:].cpu().detach().numpy(), y[j].cpu().detach().numpy()))
        L.append(ll[j].cpu().detach().numpy())
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
    x.grad=None
    for p in model.parameters():
        p.grad=None


    iter+=1
    print(modelpath + " on " + evaldata + "  flip"+str(flip) + "\n","#batch",iter, "/", maxb, " (",iter*bsize,"sams)")
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
Dict['modeltype']='iResNet(2019)'
Dict['filename']=outfile

pickle.dump(Dict,open(outfile,'wb'))

print('** Successfully accomplished!')
