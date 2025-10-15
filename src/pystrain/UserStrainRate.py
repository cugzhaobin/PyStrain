from pystrain.PyStrain import StrainRate
import numpy as np
import os

class UserStrainRate(StrainRate):
    '''
    UserStranRate is a class representing strain rate estimator at user specified points.
    '''
    
    def __init__(self, cfg):
        '''
        Constructor.
            
        Input:
            cfg = an instance of Config
        '''
        
        super(UserStrainRate, self).__init__(cfg)
        
        self.strnfile = self.usrmesh['strainfile']        
        pntsfile      = self.usrmesh['usrpntfile']
        if os.path.isfile(pntsfile) == False:
            logging.warning('The user specified point file {} does not exist'.format(pntsfile))
        else:
            self.pnts = np.genfromtxt(pntsfile, usecols=[0,1])
        self.sites = np.genfromtxt(pntsfile, usecvols=[2], dtype='str')

        
    
    def StrainRateEst(self):
        '''
        Compute strain rate at the center of each triangular mesh
        '''
        if self.usrmesh['usrsmooth']['activate'] is False:
            self.StrainRateUnSmooth()
        else:
            self.StrainRateSmooth()
    
    
    def StrainRateUnSmooth(self):
        '''
        Estimate strain rate for each triangular patch.

        '''

        distazim = Grid_Site_DistAizm(self.pnts, self.gpsvelo.llh)        
        maxdist  = self.usrmesh['maxdist']
        minsite  = self.usrmesh['minsite']
        gps      = self.gpsvelo
        print('*************')

        with open(self.strnfile, 'w') as fid:
            fid.write("#  Lon    Lat     Ve     Vn        Exx        Exy        Eyy      omega         E1         E2      shear   dilation  sec_inv_strn  theta\n")

        # each grid point
        for i in range(len(self.pnts)):
            logging.info('{} user point'.format(i))
            L   = np.ones(7)*np.NaN
            idx = np.where(np.logical_and(distazim.dist[i]<maxdist, distazim.dist[i]>1.0))[0]

            if len(idx)<minsite:
                logging.warning('{:d} GPS sites < {}km around the {:d}th grid point at {:5.2f} {:5.2f}'.format(len(idx),
                                 maxdist, i+1, self.pnts[i,0], self.pnts[i,1]))
                continue
            else:
                idx = np.argsort(distazim.dist[i])[0:minsite]
                ve  = gps.ve[idx]
                vn  = gps.vn[idx]
                se  = gps.se[idx]
                sn  = gps.sn[idx]
                gridweight = distazim.dist[i][np.argsort(distazim.dist[i])[minsite-1]]*1.5
                
            if self.grdmesh['chkazim'] == True:
                az = [0, 0, 0, 0]
                for azi in distazim.azim[i, idx]:                   
                    if azi>0 and azi < 90:
                        az[0] = 1
                    if azi>90 and azi<180:
                        az[1] = 1
                    if azi>-180 and azi<-90:
                        az[2] = 1
                    if azi>-90 and azi<0:
                        az[3] = 1
                if sum(az)<4:
                    logging.warning('The distribution of GPS sites is not reasonable for the {}th grid point at {:5.2f} {:5.2f}.'.format(
                            i+1, self.pnts[i,0], self.pnts[i,1]))
                    continue

            # debug
            if debug == True:
                print('-----------------------------------------------------')
                print('site  distance  azimuth')
                print('grid = {:d} {:8.2f} {:8.2f}'.format(i, self.pnts[i,0], self.pnts[i,1]))
                print('gridweight = {} nsite = {}'.format(gridweight, len(idx)))
                for ii in idx:
                    print('{:s} {:8.2f} {:8.2f}'.format(gps.sitename[ii], distazim.dist[i, ii], distazim.azim[i, ii]))
            
            # GPS sites < maxdist 
            x,y     = np.zeros(len(idx)), np.zeros(len(idx))
            gps_llh = gps.llh[idx]
            
            # Compute local coordinate in east and north
            for j in range(len(idx)):
                x[j], y[j] = llh2localxy(gps_llh[j], self.pnts[i])

            # Compute strain rate tensor
            L = self.strainrate(x, y, ve, vn, se, sn, gridweight)

            # Output results to file            
            with open(self.strnfile, 'a') as fid:
                fid.write('{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:8.2f}\n'.format(
                        self.pnts[i,0], self.pnts[i,1],
                        L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7],
                        L[8], L[9], L[10], L[11]))

            # Output modeled velocity
            with open('modeled.gmtvec', 'w') as fid:

        return
    
    
    def StrainRateSmooth(self):
        '''
        '''
        print('Under coding')
        pass    