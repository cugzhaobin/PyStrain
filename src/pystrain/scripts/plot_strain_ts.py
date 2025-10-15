#!/usr/bin/env python
import sys, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.interpolate import interp1d
from scipy import signal

def main(args):

    parser = argparse.ArgumentParser(description="progrom description")
    parser.add_argument('--infile', required=True)
    parser.add_argument('--size', required=False, default=5, type=int)
    parser.add_argument('--alpha', required=False, default=0.4, type=float)
    parser.add_argument('--pltdiff', required=False, default=False, type=bool)
    args   = parser.parse_args()
    size   = args.size
    infile = args.infile
    alpha  = args.alpha
    pltdiff= args.pltdiff
    
    if os.path.exists(infile) == False:
        print('The input file {} does not exist.'.format(infile))
        sys.exit()
    
    dat    = np.genfromtxt(infile)
    fig    = plt.figure(figsize=(20,14))
    
    #  No. 1 Exx
    f      = interp1d(dat[:,0], dat[:,3]*1e9)
    newt   = np.arange(min(dat[:,0]), max(dat[:,0]), 0.0027)
    ynew   = f(newt)
    b, a   = signal.butter(4, 0.008)
    newy   = signal.filtfilt(b, a, ynew)
    plt.subplot(4,2,1)
    plt.scatter(dat[:,0], dat[:,3]*1e9, s=size, c='black', alpha=alpha)
    plt.plot(newt, newy, c='gray', linewidth=3)
    if pltdiff:
        plt.scatter(dat[1:,0], np.diff(dat[:,3])*1e9, s=0.5)
    plt.ylabel('Exx (1e-9)')
    
    # No. 2 Exy
    f      = interp1d(dat[:,0], dat[:,4]*1e9)
    ynew   = f(newt)
    newy   = signal.filtfilt(b, a, ynew)
    plt.subplot(4,2,2)
    plt.scatter(dat[:,0], dat[:,4]*1e9, s=size, c='red', alpha=alpha)
    plt.plot(newt, newy, c='gray', linewidth=3)
    if pltdiff:
        plt.scatter(dat[1:,0], np.diff(dat[:,4])*1e9, s=0.5)
    plt.ylabel('Exy (1e-9)')
    
    # No. 3 Eyy
    f      = interp1d(dat[:,0], dat[:,5]*1e9)
    ynew   = f(newt)
    newy   = signal.filtfilt(b, a, ynew)
    plt.subplot(4,2,3)
    plt.scatter(dat[:,0], dat[:,5]*1e9, s=size, c='blue', alpha=alpha)
    plt.plot(newt, newy, c='gray', linewidth=3)
    if pltdiff:
        plt.scatter(dat[1:,0], np.diff(dat[:,5])*1e9, s=0.5)
    plt.ylabel('Eyy (1e-9)')
    
    # No. 4 Omega
    f      = interp1d(dat[:,0], dat[:,6]*1e9)
    ynew   = f(newt)
    newy   = signal.filtfilt(b, a, ynew)
    plt.subplot(4,2,4)
    plt.scatter(dat[:,0], dat[:,6]*1e9, s=size, c='orange', alpha=alpha)
    plt.plot(newt, newy, c='gray', linewidth=3)
    if pltdiff:
        plt.scatter(dat[1:,0], np.diff(dat[:,6])*1e9, s=0.5)
    plt.ylabel('Omega (1e-9)')
    
    # No. 5 E1
    f      = interp1d(dat[:,0], dat[:,7]*1e9)
    ynew   = f(newt)
    newy   = signal.filtfilt(b, a, ynew)
    plt.subplot(4,2,5)
    plt.scatter(dat[:,0], dat[:,7]*1e9, s=size, c='purple', alpha=alpha)
    plt.plot(newt, newy, c='gray', linewidth=3)
    if pltdiff:
        plt.scatter(dat[1:,0], np.diff(dat[:,7])*1e9, s=0.5)
    plt.ylabel('E1 (1e-9)')
    
    # No. 6 E1
    f      = interp1d(dat[:,0], dat[:,8]*1e9)
    ynew   = f(newt)
    newy   = signal.filtfilt(b, a, ynew)
    plt.subplot(4,2,6)
    plt.scatter(dat[:,0], dat[:,8]*1e9, s=size, c='darkred', alpha=alpha)
    plt.plot(newt, newy, c='gray', linewidth=3)
    if pltdiff:
        plt.scatter(dat[1:,0], np.diff(dat[:,8])*1e9, s=0.5)
    plt.ylabel('E2 (1e-9)')
    
    # No. 7 E1
    f      = interp1d(dat[:,0], dat[:,9]*1e9)
    ynew   = f(newt)
    newy   = signal.filtfilt(b, a, ynew)
    plt.subplot(4,2,7)
    plt.scatter(dat[:,0], dat[:,9]*1e9, s=size, c='seagreen', alpha=alpha)
    plt.plot(newt, newy, c='gray', linewidth=3)
    if pltdiff:
        plt.scatter(dat[1:,0], np.diff(dat[:,9])*1e9, s=0.5)
    plt.ylabel('Shear (1e-9)')
    plt.xlabel('Time (year)')
    
    # No. 8 E1
    f      = interp1d(dat[:,0], dat[:,10]*1e9)
    ynew   = f(newt)
    newy   = signal.filtfilt(b, a, ynew)
    plt.subplot(4,2,8)
    plt.scatter(dat[:,0], dat[:,10]*1e9, s=size, c='teal', alpha=alpha)
    plt.plot(newt, newy, c='gray', linewidth=3)
    if pltdiff:
        plt.scatter(dat[1:,0], np.diff(dat[:,10])*1e9, s=0.5)
    plt.ylabel('Dilation (1e-9)')
    plt.xlabel('Time (year)')
    
    plt.suptitle('{}'.format(infile[0:-4].replace('_', '-')), y=0.90)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig(infile[0:-4]+'.pdf', format='pdf', bbox_inches='tight')



    ransac = linear_model.RANSACRegressor()
    ransac.fit(dat[:,0].reshape((dat.shape[0],1)), dat[:,3])
    exxdot = ransac.estimator_.coef_[0]
    print('exxdot = {:10.2e}'.format(exxdot))
    ransac.fit(dat[:,0].reshape((dat.shape[0],1)), dat[:,4])
    exydot = ransac.estimator_.coef_[0]
    print('exydot = {:10.2e}'.format(exydot))
    ransac.fit(dat[:,0].reshape((dat.shape[0],1)), dat[:,5])
    eyydot = ransac.estimator_.coef_[0]
    print('eyydot = {:10.2e}'.format(eyydot))

    plt.show()


if __name__ == '__main__':
    main()
