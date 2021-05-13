#** Loading and running a trained ResFlow model
# most of the code is from https://github.com/rtqichen/residual-flows

import argparse
import time
import math
import os
import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets
from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim
import lib.utils as utils
import lib.layers as layers
import lib.layers.base as base_layers
import sys


def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x
def update_lipschitz(model):
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            m.compute_weight(update=True)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            m.compute_weight(update=True)
def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', type=str, default='cifar10', choices=[
        'mnist',
        'cifar10',
        'svhn',
        'celebahq',
        'celeba_5bit',
        'imagenet32',
        'imagenet64',
    ]
)
parser.add_argument('--dataroot', type=str, default='data')
parser.add_argument('--imagesize', type=int, default=32)
parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq.

parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')
parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)

parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False)
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='poisson')
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-exact-terms', type=int, default=2)
parser.add_argument('--var-reduc-lr', type=float, default=0)
parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)

parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='swish')
parser.add_argument('--idim', type=int, default=512)
parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')
parser.add_argument('--add-noise', type=eval, choices=[True, False], default=True)
parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
parser.add_argument('--fc-idim', type=int, default=128)
parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)
parser.add_argument('--cdim', type=int, default=256)

parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)
parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--annealing-iters', type=int, default=0)
parser.add_argument('--save', help='directory to save results', type=str, default='experiment1')
parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid'], default='density')
parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--begin-epoch', type=int, default=0)

parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=20)
parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=500)

args = parser.parse_args(['--optimizer','adam'])



device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

if args.squeeze_first:
    squeeze_layer = layers.SqueezeLayer(2)

print('Creating model.', flush=True)


def LoadResflow(path,dataset):

    global input_size
    global args

    if dataset=='mnist':
        args.imagesize = 28
        args.actnorm = True
    if dataset=='imagenet64':
        args.nblocks='32-32-32'
        args.actnorm = True
        args.squeeze_first=True
        args.factor_out=True
    if dataset=='celeba_5bit':
        args.imagesize = 64
        args.actnorm = False
        args.nbits=5
        args.act='elu'
        args.nblocks='16-16-16-16'
        args.squeeze_first=True
        args.factor_out=True
        args.n_exact_terms=8
        args.fc_end=False


    if dataset == 'cifar10':
        im_dim = 3
        n_classes = 10
        if args.task in ['classification', 'hybrid']:

            # Classification-specific preprocessing.

            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor()
            ])

            # Remove the logit transform.
            init_layer = layers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor()
            ])
            init_layer = layers.LogitTransform(0.05)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataroot+"/cifar10", train=False, transform=transform_test),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
    elif dataset == 'mnist':
        im_dim = 1
        init_layer = layers.LogitTransform(1e-6)
        n_classes = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.dataroot+"/mnist", train=True, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor()
                ])
            ),
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.dataroot+"/mnist", train=False, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor()
                ])
            ),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
    elif dataset == 'svhn':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        n_classes = 10
        test_loader = torch.utils.data.DataLoader(
            vdsets.SVHN(
                args.dataroot+"/svhn", split='test', download=True, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor()
                ])
            ),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
    elif dataset == 'celebahq':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        if args.imagesize != 256:
            args.imagesize = 256

        test_loader = torch.utils.data.DataLoader(
            datasets.CelebAHQ(
                train=False, transform=transforms.Compose([
                    reduce_bits,
                ])
            ), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )
    elif dataset == 'celeba_5bit':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        if args.imagesize != 64:
            args.imagesize = 64

        test_loader = torch.utils.data.DataLoader(
            datasets.CelebA5bit(train=False, transform=transforms.Compose([
            ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )
    elif dataset == 'imagenet32':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        if args.imagesize != 32:
            args.imagesize = 32

        test_loader = torch.utils.data.DataLoader(
            datasets.Imagenet32(train=False, transform=transforms.Compose([
            ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )
    elif dataset == 'imagenet64':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        if args.imagesize != 64:
            args.imagesize = 64

        test_loader = torch.utils.data.DataLoader(
            datasets.Imagenet64(train=False, transform=transforms.Compose([

            ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )

    print('Dataset loaded.', flush=True)

    input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)

    if args.squeeze_first:
        input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)

    # Model
    model = ResidualFlow(
        input_size,
        n_blocks=list(map(int, args.nblocks.split('-'))),
        intermediate_dim=args.idim,
        factor_out=args.factor_out,
        quadratic=args.quadratic,
        init_layer=init_layer,
        actnorm=args.actnorm,
        fc_actnorm=args.fc_actnorm,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        fc=args.fc,
        coeff=args.coeff,
        vnorms=args.vnorms,
        n_lipschitz_iters=args.n_lipschitz_iters,
        sn_atol=args.sn_tol,
        sn_rtol=args.sn_tol,
        n_power_series=args.n_power_series,
        n_dist=args.n_dist,
        n_samples=args.n_samples,
        kernels=args.kernels,
        activation_fn=args.act,
        fc_end=args.fc_end,
        fc_idim=args.fc_idim,
        n_exact_terms=args.n_exact_terms,
        preact=args.preact,
        neumann_grad=args.neumann_grad,
        grad_in_forward=args.mem_eff,
        first_resblock=args.first_resblock,
        learn_p=args.learn_p,
        block_type='resblock',
    )

    model.to(device)

    print('Initializing model.', flush=True)

    with torch.no_grad():
        x = torch.rand(1, *input_size[1:]).to(device)
        model(x)
    print('Restoring from checkpoint.', flush=True)
    checkpt = torch.load(path)

    sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
    state = model.state_dict()
    state.update(sd)
    model.load_state_dict(state, strict=True)

    #state = model.state_dict()
    #model.load_state_dict(checkpt['state_dict'], strict=True)

    ema = utils.ExponentialMovingAverage(model)
    ema.set(checkpt['ema'])
    ema.swap()

    print(model, flush=True)

    model.eval()
    print('Updating lipschitz.', flush=True)
    update_lipschitz(model)

    return model, test_loader

def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)

def add_padding(x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)

def Forward(x, model, computell=True):

    x = x.to(device)

    if not computell:
        return model(x, None),None

    im_dim = 3

    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256

    x, logpu = add_padding(x, nvals)

    if args.squeeze_first:
        x = squeeze_layer(x)

    z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

    # log p(z)
    logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

    # log p(x)
    logpx = logpz - delta_logp - np.log(nvals) * (args.imagesize * args.imagesize * (im_dim + args.padding)) - logpu

    return z, logpx


def Backward(z,model):
    return model.inverse(z,None)


def GetData(dataset,trainset=False):

    if dataset=='mnist':
        args.imagesize = 28
        args.actnorm = True

    if dataset == 'cifar10':
        if args.task in ['classification', 'hybrid']:

            # Classification-specific preprocessing.

            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor()
            ])

            # Remove the logit transform.
            init_layer = layers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor()
            ])
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataroot+"/cifar10", train=trainset, transform=transform_test),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
    elif dataset == 'cifar100':
        if args.task in ['classification', 'hybrid']:

            # Classification-specific preprocessing.

            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor()
            ])

            # Remove the logit transform.
            init_layer = layers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor()
            ])
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.dataroot+"/cifar100", train=trainset, transform=transform_test),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
    elif dataset == 'mnist':
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.dataroot+"/mnist", train=trainset, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor()
                ])
            ),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
    elif dataset == 'fashion':
        test_loader = torch.utils.data.DataLoader(
            datasets.FASHION(
                args.dataroot+"/fashion", train=trainset, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor()
                ])
            ),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
    elif dataset == 'svhn' and trainset==False:
        test_loader = torch.utils.data.DataLoader(
            vdsets.SVHN(
                args.dataroot+"/svhn", split='test', download=True, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor()
                ])
            ),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
    elif dataset == 'celebahq':
        test_loader = torch.utils.data.DataLoader(
            datasets.CelebAHQ(
                train=trainset, transform=transforms.Compose([
                    reduce_bits,
                ])
            ), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )
    elif dataset == 'celeba_5bit':
        test_loader = torch.utils.data.DataLoader(
            datasets.CelebA5bit(train=trainset, transform=transforms.Compose([
            ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )
    elif dataset == 'imagenet32':
        test_loader = torch.utils.data.DataLoader(
            datasets.Imagenet32(train=trainset, transform=transforms.Compose([
            ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )
    elif dataset == 'imagenet64':
        test_loader = torch.utils.data.DataLoader(
            datasets.Imagenet64(train=trainset, transform=transforms.Compose([

            ])),batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )

    return test_loader

def Blankloader():
    while True:
        r=torch.rand(1)
        im=(torch.ones([args.batchsize,3,args.imagesize,args.imagesize])*r)
        yield (im,torch.tensor(0))

def Whitenoiseloader():
    while True:
        r=torch.randn([args.batchsize,3,args.imagesize,args.imagesize])
        yield (r,torch.tensor(0))

def Uninoiseloader():
    while True:
        r=torch.rand([args.batchsize,3,args.imagesize,args.imagesize])
        yield (r,torch.tensor(0))
