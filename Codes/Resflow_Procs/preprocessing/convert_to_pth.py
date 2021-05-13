import sys
import re
import numpy as np
import torch

infile='celeba_full_64x64_5bit.npy'
img = torch.tensor(np.load(infile))
img = img.permute(0, 3, 1, 2)
torch.save(img, re.sub('.npy$', '.pth', infile))
