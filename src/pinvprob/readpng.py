#!/usr/bin/python
import pylab
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_bwimg(file):
    bwimg=mpimg.imread(file)
    bwimg=np.sum(bwimg,axis=2)/3.0    
    return bwimg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', nargs=1, required=True, help='png file')
    args = parser.parse_args()    
    img=get_bwimg(args.f[0])
    print(img.shape)    
    fig =plt.figure()
    imshow(img,cmap="gray")
    plt.show()
