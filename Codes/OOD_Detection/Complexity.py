#** Computing the complexity related statistic (the size of the compressed image), and appending it to the statistics data file
# (Requires that the FLIF image compressor is installed on the system)

import os
import torch
import pickle
import torchvision.utils as utils
import argparse


#bias that must be added to image pixels so that the range becomes [0,1]. use 0.5 for IresNet and Glow, and 0 for ResFlow
imbias = 0
rpath='/home/InvOOD/OodClassifier/' #path of this script

def ComputeImageCompx(img):
    img=torch.tensor(img)
    #saving the image and running the FLIF compressor on it
    utils.save_image(img,"temp_im.png")
    os.system('flif -e --overwrite ' + rpath +'temp_im.png '+rpath +'temp_im.flif')
    nbytes=os.path.getsize(rpath +'temp_im.flif') #getting the output image file size

    return (nbytes*8)

# reading an existing statistics data file, computing the complexity measure of each image, and appending the values to the file
def AppendComptoFile(file):

    print(file)

    with open(file, 'rb') as f:
        indic = pickle.load(f)

    data = indic['data']['datapoints']

    cmps = []
    i = 0

    print("computing complexities...")
    for x, _ in data:
        img = x+imbias

        c = ComputeImageCompx(img)
        cmps.append(c)

        i += 1

    indic['data']['compxs'] = cmps

    pickle.dump(indic, open(file, 'bw'))
    print('written to file.')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',type=str,help="path of the statistics file to which the complexity statistic will be appended") 
    args=parser.parse_args()
    AppendComptoFile(args.input)

