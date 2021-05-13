# ** Computing all the statistics useful for OOD detection and saving them to disk, given
# the outputs (latent codes, likelihood, derivatives) of a generative (invertible) model on a dataset
# The implemented statistics are:
#   LL: log-likelihood
#   LPZ: prior log-density, i.e log{p(z)} (T_1)
#   EJMA: mean of absolute value of encoder Jacobian column-sum (T_3)
#   LDET: Jacobian log-determinant term
#   CDFD: histogram distance between the empirical distribution of the latent variables and standard Gaussian
#   PWABS: the average of absolute pairwise products between latent dimesnions i.e. |z_i||z_j| for i != j
#   GNP: the norm of log-likelihood gradient w.r.t parameters
# Note that the complexity based statistic ('compxs'=T_2) should be computed using the script 'complexity.py'


import mystats
import numpy as np
import pickle
import sys
import argparse



to_compute={'LPZ','EJMA'} # the set of the statistics to compute. see the definitions in the header

parser = argparse.ArgumentParser()
parser.add_argument('input',type=str,help="path of the input file which contains the results of running a generative model") 
parser.add_argument('output',type=str,help="the path of the output file for writing the statistics values")
parser.add_argument('-update',default=False,help="if the new results should be appended to an existing output file",action='store_true')
args=parser.parse_args()

fname=args.input
infile=args.input
outfile=args.output
update=args.update
print("Obtaining the following statistics:",to_compute) 


Dic=pickle.load(open(infile,'rb'))
print("file loaded: ",infile)
nsamples=Dic['ndata']
print("#samples: ",nsamples)


vals=dict()
if update:
    print(">update mode")
    vals=pickle.load(open(outfile,'rb'))

for k in to_compute:
    vals[k]=[]



if 'LL' in to_compute:
    print("saving loglikelihoods...")
    for l in Dic['LL']:
        vals['LL'].append(l)

    print(len(vals['LL']),"values written")


if 'GNP' in to_compute:
    print("saving the norm of LL gradient w.r.t params...")
    for v in Dic['GNParam']:
        vals['GNP'].append(v)

i = 0
print("computing latent-based stats...")
for z in Dic['Z']:

    i += 1
    if i%100==0:
        print(int(i*100/float(nsamples)),"%")

    zd=z.flatten()

    if 'LPZ' in to_compute: # prior logprobability, i.e log[p(z)]
        vals['LPZ'].append(mystats.logpdf(zd))

    if 'CDFD' in to_compute: # histogram distance between the empirical distribution of the latent variables and standard Gaussian
        vals['CDFD'].append(mystats.cdfdif(zd))

    if 'PWABS' in to_compute: # the average of absolute pairwise products between latent dimesnions i.e. |z_i||z_j| for i != j
        vals['PWABS'].append(mystats.pairwiseabs(zd))


if 'EJMA' in to_compute:
    i=0
    print("computing the mean of absolute value of encoder Jacobian column-sum ...")
    for g in Dic['Encoder_Jac']:
        gm=np.mean(np.abs(g))
        vals['EJMA'].append(gm)
        i+=1
        if i%100==0:
            print(int(float(i) * 100 / nsamples), "%")


if 'LDET' in to_compute:
    i=0

    if not ('LL' in to_compute) or not ('LPZ' in to_compute):
        raise Exception('needs LL and LPZ')

    print("computing log-det terms...")
    for i in range(0,nsamples):
        vals['LDET'].append(vals['LL'][i]-vals['LPZ'][i])
        i+=1
        if i%100==0:
            print(int(float(i)*100/nsamples), "%")


vals['datapoints']=Dic['Data']

# writing the results and some information to disk
rdic= dict()
rdic['data']=vals
rdic['stats']=to_compute
rdic['nsamples']=nsamples
rdic['info']=Dic['info']
rdic['modeltype']=Dic['modeltype']
rdic['filename']=outfile

pickle.dump(rdic,open(outfile,'bw'))
print("** successfully dumped to file!", ":",outfile)
