#** Computing Glow outputs on a dataset, which are latent codes, likelihood, Jacobian column sum, and gradient wrt prameters,
#     and saving them to a file, so that OOD detection statistics can be later obtained from them

import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)
import datasets
from model import Glow
from os.path import getmtime
import torch.autograd as autograd
import pickle
import json
import argparse

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-modelpath',type=str,help='path of the pytorch trained model directory') 
parser.add_argument('-trainedon',type=str,default='cifar10_train',help='dataset name on which the model has been trained')
parser.add_argument('-teston',type=str,help="dataset name on which the model outputs/statistics are computed; one of {'cifar10_test','cifar100_test','svhn_test'}")
parser.add_argument('-flipmode',type=int,default=0,help='whether to flip the test images (0:disabled, 1:vertical, 2:horizontal)')
parser.add_argument('-n',type=int,help='number of test samples (images) to process')
parser.add_argument('-outfile',type=str,help='file name to write the output')
args=parser.parse_args()

model_date='pretrained'
bsize=32 # batch size
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #CUDA device
#--


modelpath=args.modelpath
model_trained_data=args.trainedon
evaldata=args.teston
flip=args.flipmode
outfile=args.outfile
nsamples=args.n

maxb=nsamples//bsize #number of batches to process


print("batchsize:",bsize,"device:",device)

# loading the data and model

if evaldata=='cifar10_test':
    image_shape, num_classes, _,testset=datasets.get_CIFAR10(False,".", True)
    dataset=torch.utils.data.DataLoader(testset, batch_size=bsize, num_workers=4)
if evaldata=='cifar100_test':
    image_shape, num_classes, _,testset=datasets.get_CIFAR100(False,".", True)
    dataset=torch.utils.data.DataLoader(testset, batch_size=bsize, num_workers=4)
if evaldata=='svhn_test':
    image_shape, num_classes, _,testset=datasets.get_SVHN(False,".", True)
    dataset = torch.utils.data.DataLoader(testset, batch_size=bsize, num_workers=4)

model_name = 'glow_affine_coupling.pt'
with open(modelpath + 'hparams.json') as json_file:
    hparams = json.load(json_file)

model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(modelpath + model_name, map_location=device))
model.set_actnorm_init()
model.to(device)
model = model.eval()

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
print("model loaded. #parameters:",npr)

# the main loop for computing model outputs, for each batch
iter=0

for (x,y) in dataset:

    # apply flipping if enabled
    if flip == 1:
        x = x.flip(2)
    if flip == 2:
        x = x.flip(3)
    # move tensor to GPU (if available) and enable gradient computation wrt input images
    x = x.to(device)
    x.requires_grad_()

    # forward propagation (computing latent code and log-likelihood)
    zs, ll = model(x,return_latent_mode='full')

    imsize=x.shape[1]*x.shape[2]*x.shape[3]

    # the latent code returned by iResnet is arranged in a multiscale structure, but we need to flatten it
    z=torch.zeros((bsize,1,imsize)) #an array for storing the latent vectors of the batch, one sample per row
    for k in range(bsize):
        z[k,:] = torch.cat((zs[0][k], zs[1][k], zs[2][k])) # concatenating all dimensions of latent codes

    # computing the column sum of the encoder's Jacobian matrix, i.e vJ where v is a vector of ones,
    # or equivalently gradient of f(z)=z_1+...+z_m wrt to x
    jr = autograd.grad(z, x, grad_outputs=[torch.ones_like(z)], only_inputs=True, retain_graph=True)

    for j in range(0, bsize):

        # appending the obtained values for each sample in the batch to the corresponding arrays
        D.append((x[j,:].cpu().detach().numpy(),y[j].cpu().detach().numpy()))
        Z.append(z[j, 0, :].cpu().detach().numpy())
        L.append(ll[j].cpu().detach().numpy().flatten()[0])
        En_Jac.append(jr[0][j, :].cpu().detach().numpy())

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


    iter += 1
    print("glow " + model_trained_data + " on " + evaldata + "\n", "#batch", iter, "/", maxb, " (", iter * bsize, "sams)")
    if iter == maxb:
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

pickle.dump(Dict,open(outfile,'wb'))
