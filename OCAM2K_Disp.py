#!/usr/bin/env kpython3
import getopt
import sys
import time
import pysao
import OCAM2K_Images
import numpy as np
## import library used to manipulate fits files
from astropy.io import fits

try:
    bkgd = fits.open('/home/aodev/Data/220201/OCAM2/bkg_g300_med.fits')[0].data
    print('bkgd loaded')
except:
#if True:
    bkgd = np.zeros([240,240])


if __name__ == '__main__':
    ds9=pysao.ds9()
    if len(sys.argv) == 1:
        inter = 0
    else:
        inter = int(sys.argv[1])
    while True:
        if inter:
            input()
        ds9.view(OCAM2K_Images.get_image()[0].astype(np.float32)-bkgd)
        time.sleep(0.1)
