#** Some of the statistics computed from latent codes

import scipy.stats
import numpy as np
import torch



#LPZ
def logpdf(z):

    lpdf=scipy.stats.norm.logpdf

    lls=lpdf(z)
    ll=np.sum(lls)

    return ll

#CDFD
def cdfdif(z):

    z.sort()

    dif = 0
    n=z.size

    for k in range(0, n):
        y1 = float(k+1) / n
        y2=scipy.stats.norm(loc=0.0,scale=1.0).cdf(z[k])
        dif += abs(y1 - y2)

    return (dif / n)

def cdfdifmax(z):

    z.sort()

    dif = 0
    n=z.size

    for k in range(0, n):
        y1 = float(k+1) / n
        y2=scipy.stats.norm(loc=0.0,scale=1.0).cdf(z[k])
        dif = max(abs(y1 - y2),dif)

    return dif

#PWABS
def pairwiseabs(z):
    n = z.size
    zm = z[:, np.newaxis]
    c = np.matmul(zm, zm.transpose())
    c = np.abs(c)
    n = (n** 2 - n) / 2

    return (c.sum() - c.trace()) / n

#PW
def pairwise(z):
    n = z.size
    zm = z[:, np.newaxis]
    c = np.matmul(zm, zm.transpose())
    n = (n** 2 - n) / 2

    return (c.sum() - c.trace()) / n

#PWABS
def pairwiseabs_gpu(z):
    n = z.size
    zm = z[:, np.newaxis]
    zm=torch.tensor(zm,device=dev)
    c = torch.matmul(zm, zm.transpose(0,1))

    c = torch.abs(c)
    c=c.cpu().numpy()
    n = (n** 2 - n) / 2

    return (c.sum() - c.trace()) / n

