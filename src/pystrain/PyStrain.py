"""
Created on Fri Apr 10 11:08:15 2020

@author: zhao
"""
__all__=['PosData', 'GPSVelo', 'Config', 'Strain', 'GridPoint', 'StrainRate', 
         'GrdStrainRate', 'TriStrainRate', 'UserStrainRate', 'GridPoint']

import os, sys
import numpy as np
from pyproj import Geod, Proj
import yaml, logging, glob
import matplotlib.tri as mtri
from sklearn import linear_model
import GPSTime as gpstime
debug=False
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'
                    )

def polyconic(Lat, Diff_long, Lat_Orig):
    p1 = Lat_Orig
    p2 = Lat
    il = Diff_long

    arcone = 4.8481368e-6
    esq = 6.7686580e-3
    la = 6378206.4
    a0 = 6367399.7
    a2 = 32433.888
    a4 = 34.4187
    a6 = .0454
    a8 = 6.0e-5

    ip = p2 - p1
    sinp2 = np.sin(p2 * arcone)
    cosp2 = np.cos(p2 * arcone)
    theta = il * sinp2
    a = np.sqrt(1.0 - (esq * (2. * sinp2))) / (la * arcone)
    cot = cosp2 / sinp2
    x = (cot * np.sin(theta * arcone)) / (a * arcone)
    ipr = ip *arcone
    pr = ((p2 + p1) /2.) * arcone
    y = ((((a0*ipr) - ((a2*np.cos(2.*pr))*np.sin(ipr))) + ((a4*np.cos(4.*pr))*np.sin(2.*ipr))) 
            - ((a6*np.cos(6.*pr))*np.sin(3.*ipr))) + ((a8*np.cos(8.*pr))*np.sin(4.*ipr))
    xy = np.array([x, y])
 
    return xy


def llh2localxy(llh, ll_org):
    '''
    convert coordinate from lat lon to XY
    
    Input:
      llh    - lon, lat
      ll_org - refernence point (lon, lat)
    Output:
      xy     - east(km), north(km)
     
    '''

 
    lat = 3600*llh[1]
    lon = 3600*llh[0]
    
    Lat_Orig  = 3600.0 * ll_org[1]
    Diff_long = 3600.0 * ll_org[0] - lon
    xy = polyconic(lat, Diff_long, Lat_Orig)

    xy[0] = -xy[0]/1000.0
    xy[1] =  xy[1]/1000.0

    return xy

def llh2utm(llh, llh_org):
     '''
     Convert coordinate from [lon, lat] to [east, north] w.r.t original coordinate

     Input:
         llh    = lon, lat in degree
         ll_org = refernence point (lon, lat)
     Output:
         e_loc  = east(km)
         n_loc  = north(km)

     '''
     P      = Proj(ellps='WGS84', proj='tmerc', lon_0=llh_org[0])
     en     = P(llh[0], llh[1])
     en_org = P(llh_org[0], llh_org[1])
     e_loc  = (en[0]-en_org[0])/1e3
     n_loc  = (en[1]-en_org[1])/1e3

     return [e_loc, n_loc]  


class Config(object):
    '''
    Config is a class representing configure file used for strain estimation.
    '''
    def __init__(self, config_file):
        '''
        Constructor.
        
        Input:
            config_file = YAML format file
        '''
        
        if os.path.isfile(config_file) is False:
            logging.critical(' configure file {} does not exit!'.format(config_file))
            sys.exit(0)
            
        self.config_file = config_file
        self._parse_configure()
        logging.debug('Configure file {} is parsed'.format(config_file))
        
    def _parse_configure(self):
        '''
        parse configure file using yaml module
        '''
        with open(self.config_file, 'r') as fid:
            lines = fid.read()
            cfg   = yaml.load(lines, Loader=yaml.FullLoader)
            self.cfg = cfg


class PosData(object):
    '''
    POSDATA is a class representing a PBO POS file.
    '''

    
    def __init__(self, posfile):
        '''
        Constructor.
        Mod by Zhao Bin, Jan. 10, 2019. Fix bug when reading pos file
        Mod by Zhao Bin, Jan. 11, 2019. Fix bug when reading pos file, use delimiter to foramt the file
        
        Input:
            posfile    = file name of a pos file
        '''
        
        # station ID in 4 char
        self.site = ""
        
        # MJD
        self.MJD  = []
        
        # Decimal yar
        self.decyr= []
        
        # North displacement in meter
        self.N    = []
        
        # East displacement in meter
        self.E    = []
        
        # Up displacement in meter
        self.U    = []
        
        # North uncertainty in meter
        self.SN   = []
        
        # East uncertainty in meter
        self.SE   = []
        
        # Up uncertainty in meter
        self.SU   = []
        
        self._LoadPOSData(posfile)
        
        

    def _LoadPOSData(self, posfile):
        # check the input file exists
        if os.path.isfile(posfile) is False:
            logging.warning(' The time series file {:s} does not exist'.format(posfile))
            sys.exit()
            
        # open the file and read the header
        with open(posfile) as fid:
            for line in fid:
                if line[0] != '':
                    if line.find('ID') != -1:
                        self.site = line[16:20]
                    if line.find('NEU') == 0:
                        self.lat = float(line.split()[4])
                        self.lon = float(line.split()[5])
                        self.hei = float(line.split()[6].replace('*','0'))
                elif line[0] == ' ':
                    continue
        
        # read the time series and convert to millimeter
        data     = np.genfromtxt(posfile, skip_header=37, \
                   delimiter=(9,7,11,15,15,15,9,9,9,7,7,7,19,16,11,12,10,10,11,9,9,7,7,7,6))
        if data.ndim == 1:
            data = data.reshape((1, len(data)))
        idx      = list(set(range(len(data))) - set(np.where(np.isnan(data[:,[15,16,17]]))[0]))
        self.MJD = data[idx,2]
        self.NLat= data[idx,12]
        self.ELon= data[idx,13]
        self.N   = data[idx,15]*1e3
        self.E   = data[idx,16]*1e3
        self.U   = data[idx,17]*1e3
        self.SN  = data[idx,18]*1e3
        self.SE  = data[idx,19]*1e3
        self.SU  = data[idx,20]*1e3
        
        # convert the MJD to decimal year
        self.decyr = np.array([gpstime.jd_to_decyrs(self.MJD[i]) 
                                for i in range(len(self.MJD))])
        

class IOSData(object):
    '''
    IOSData is a class of GPS time series output from PyTsfit with correction for offsets/breaks and 
    seasonal variations.
    '''
    
    def __init__(self, neufile):
        '''
        '''
        self.site  = neufile[0:4]
        self._LoadData(neufile)
        
        
    def _LoadData(self, neufile, scale=1.0):
        # check the input file exists
        if os.path.isfile(neufile) is False:
            logging.warning(' The time series file {:s} does not exist'.format(neufile))
            sys.exit()
            
        # read the time series and convert to millimeter
        data     = np.genfromtxt(neufile)
        if data.ndim == 1:
            data = data.reshape((1, len(data)))
        idx      = list(set(range(len(data))) - set(np.where(np.isnan(data[:,[1,2,3]]))[0]))
        self.decyr = data[idx,0]
        self.N     = data[idx,1]*scale
        self.E     = data[idx,2]*scale
        self.U     = data[idx,3]*scale
        self.SN    = data[idx,4]*scale
        self.SE    = data[idx,5]*scale
        self.SU    = data[idx,6]*scale


class GPSVelo(object):
    '''
    GPSVelo is a class of GPS velocity used to infer strain rate.
    '''
    
    def __init__(self, velofile, velotype):
        '''
        Constructor.
        
        Input:
            velofile = GPS velocity file
            velotype = GMT/GLOBK
        '''
        if os.path.isfile(velofile) is False:
            logging.critical('Input velocity file {:s} does not exit'.format(velofile))
            sys.exit(0)
        
        self._LoadVeloData(velofile, velotype)
        logging.info('{} velocities from {} are loaded'.format(self.SiteNum, velofile))
            
    
    def _LoadVeloData(self, velofile, velotype):
        '''
        Load velocity from input data
        '''
        if velotype.upper() == 'GMT':
            velo = np.genfromtxt(velofile, comments='#')
            self.llh = velo[:,[0,1]]
            self.ve  = velo[:,2]
            self.vn  = velo[:,3]
            self.se  = velo[:,4]
            self.sn  = velo[:,5]
            
            # Read site name
            if velo.shape[1] == 8:
                self.sitename = np.genfromtxt(velofile, comments='#', 
                                              usecols=[7], dtype='str')
            else:
                self.sitename = None
                    
        elif velotype.upper() == 'GLOBK':
            velo = np.genfromtxt(velofile, comments='#')
            self.llh = velo[:,[0,1]]
            self.ve  = velo[:,2]
            self.vn  = velo[:,5]
            self.se  = velo[:,4]
            self.sn  = velo[:,7]
            
            #
            if velo.shape[1] == 9:
                self.sitename = np.genfromtxt(velofile, comments='#', 
                                              usecols=[9], dtype='S')
            else:
                self.sitename = None  
        return
    
    @property
    def SiteName(self):
        return self.sitename
    
    @property
    def SiteNum(self):
        return len(self.llh)
    
    @property
    def VeloData(self):
        return (self.ve, self.vn)
    
    @property
    def VeloSigma(self):
        return (self.se, self.sn)

class GridPoint(object):
    '''
    GridPoint is a class representing grid point.
    '''
    def __init__(self, slon, elon, slat, elat, dn, de):
        '''
        Constructor.
        
        Input:
            slon, elon = longitude of start and end points
            slat, elat = latitude of start and end points
            dn, de     = delta n and delta e
        Ouput:
            
        '''
        if slon > elon:
            slon, elon = elon, slon
        if slat > elat:
            slat, elat = elat, slat
                
        lon = np.arange(slon, elon, dn)
        lat = np.arange(slat, elat, de)
        idx = np.arange(0, len(lon), 2)

        grd            = np.meshgrid(lon, lat)
        grd_llh        = np.vstack((grd[0].flatten(), grd[1].flatten())).T
        grd_llh[idx,:] = grd_llh[idx,:]+(elon-slon)/len(lon)/2
        self.llh       = grd_llh
        logging.debug('{} grid points are created'.format(len(grd_llh)))



class Strain(object):
    '''
    Strain is class representing strain
    '''
    
    def __init__(self, cfg):
        '''
        '''
        pass        

    
    @staticmethod
    def strainrate(x, y, ve, vn, se, sn, gridweight=None):
        '''
        Compute strain rate.
        
        Input:
            x, y       = local coordinates in east and north direction from grid
                         point to GPS sites in kilometer
            ve, vn     = velocities in east and north directions in mm/yr
            se, sn     = uncertainties of velocities in east and north directions in mm/yr
            gridweight = distance R0, if None means uniform weight of 1.
        Output:
        dx,dy,exx,exy,eyy,w,E1,E2,gamma,delta,theta
            dx         = east velocity
            dy         = north velocity
            exx,exy,eyy= strain rate tensor
            w          = rotation rate
            E1         = Maximum strain rate
            E2         = Minimum strain rate
            gamma      = Maximum shear strain rate
            delta      = dialtion strain rate
            theta      = azimuth of Maximum strain rate 
        '''
        nsite = len(x)
        G     = np.zeros((2*nsite,6))
        U     = np.zeros((2*nsite,1))
        W     = np.zeros((2*nsite,1))
        D     = np.sqrt(np.power(x,2)+np.power(y,2))
        
        for i in range(nsite):
            U[2*i,0]   = ve[i]
            U[2*i+1,0] = vn[i]    
            if gridweight is None:
                w      = 1.0
            else:
                w      = np.exp(D[i]**2/gridweight**2)

            # East
            G[2*i,0]   = 1
            G[2*i,1]   = 0
            G[2*i,2]   = x[i]/1e3
            G[2*i,3]   = y[i]/1e3
            G[2*i,4]   = 0
            G[2*i,5]   = y[i]/1e3

            # North 
            G[2*i+1,0] = 0
            G[2*i+1,1] = 1
            G[2*i+1,2] = 0
            G[2*i+1,3] = x[i]/1e3
            G[2*i+1,4] = y[i]/1e3
            G[2*i+1,5] =-x[i]/1e3
            
            W[2*i,0]   = 1/(se[i]*w)**2
            W[2*i+1,0] = 1/(sn[i]*w)**2
        
#       L = np.linalg.lstsq(np.multiply(G,W), np.multiply(W,U), rcond=None)[0]
#       from scipy.linalg import lstsq
#       L,res,rnk,s = lstsq(np.multiply(G,W), np.multiply(W,U), cond=None)
#       print(L)
        
        ATPA = np.multiply(G.T, W.T).dot(G)
        ATPL = np.multiply(G.T, W.T).dot(U)
        L  = np.linalg.pinv(ATPA).dot(ATPL)

        dx  = L[0,0]
        dy  = L[1,0]
        exx = L[2,0]*1e-9
        exy = L[3,0]*1e-9
        eyy = L[4,0]*1e-9
        w   = L[5,0]*1e-9
        if debug == True:
            v = np.column_stack((ve, vn)).flatten()
            for i in np.arange(len(G)):
                print('{} {} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f}'.
                        format(G[i,0], G[i,1], G[i,2], G[i,3], G[i,4], G[i,5], v[i], W[i,0]))
        
        # dilational strain (rate)
        delta  = exx+eyy
        gamma1 = eyy-exx
        gamma2 = 2*exy

        # maximum shear strain (rate)
        gamma  = np.sqrt(gamma1**2 + gamma2**2)

        # maximum principle strain (rate)
        E1     = (delta+gamma)/2
        # miniumum principle stran (rate)
        E2     = (delta-gamma)/2
        
        # direction of maximum strain (rate)
        theta = 1/2.0*np.arctan2(2.0*exy,gamma1)
        theta = np.rad2deg(theta)
        if gamma1 < 0: theta = theta + 90
        if gamma2 < 0 and gamma1 > 0:
            theta = theta + 180

        if theta < 90:
            theta = theta+270
        else:
            theta = theta-90

        theta = theta-90
        # second invariant of the strain rate
        sec_inv = np.sqrt(exx**2+2*exy**2+eyy**2)
        
        return (dx,dy,exx,exy,eyy,w,E1,E2,gamma,delta,sec_inv, theta)
    
    
    
class Grid_Site_DistAizm(object):
    '''
    Grid_Site_DistAizm is a class representing distance and azimuth between grid
    point and GPS stations.
    '''
    def __init__(self, grd_llh, gps_llh):
        '''
        Constructor
        
        Input:
            grd_llh = [lon, lat] 
            gps_llh = [lon, lat]
        '''
        self.grd_llh = grd_llh
        self.gps_llh = gps_llh

        self._distazim()
        logging.debug('Distance and azimuth from {} grid points to {} GPS sites are computed.'
                      .format(len(grd_llh), len(gps_llh)))
        
        
    def _distazim(self):
        '''
        Compute distance and azimuth between grid points and GPS stations using pyproj module.
        '''
        
        grd_llh = self.grd_llh
        gps_llh = self.gps_llh
        
        geod = Geod(ellps='WGS84')
        dist = np.zeros((len(grd_llh), len(gps_llh)))
        azim = np.copy(dist)
        for i in range(len(grd_llh)):
            for j in range(len(gps_llh)):
                v = geod.inv(grd_llh[i,0], grd_llh[i,1], gps_llh[j,0], gps_llh[j,1])
                dist[i,j] = v[2]
                azim[i,j] = v[0]
        
        # Convert to km
        self.dist = dist/1e3
        self.azim = azim
        return        


class StrainRate(Strain):
    '''
    StaticStrain is a class representing strain estimator from GPS velocity.
    '''
    
    def __init__(self, cfg):
        '''
        Constrctor.
        
        Input:
            cfg = an instance of Config
        '''
        
        self.cfg     = cfg['strain_rate']
        gpsvelo      = self.cfg['gpsvelo']
        velofmt      = self.cfg['velofmt']
        grdmesh      = self.cfg['grdmesh']
        trimesh      = self.cfg['trimesh']
        usrmesh      = self.cfg['usrmesh']
        
        self.gpsvelo  = GPSVelo(gpsvelo, velofmt)
        self.grdmesh  = grdmesh
        self.trimesh  = trimesh
        self.usrmesh  = usrmesh
    
    def StrainRateEst(self):
        pass
    
    def StrainRateUnSmooth(self):
        pass
    
    def StrainRateSmooth(self):
        pass
        
        
class GrdStrainRate(StrainRate):
    '''
    GrdStrainRate is a class representing strain rate estimator for grid points
    '''
    
    def __init__(self, cfg):
        '''
        Constructor.
            
        Input:
            cfg = an instance of Config
        '''
        super(GrdStrainRate, self).__init__(cfg)        
        grdmesh       = self.grdmesh
        self.strnfile = grdmesh['strainfile']
        self.gdp      = GridPoint(grdmesh['slon'], grdmesh['elon'],\
                                  grdmesh['slat'], grdmesh['elat'],\
                                  grdmesh['dn']  , grdmesh['de'])
        
    
    def StrainRateEst(self):
        '''
        Estimate strain rate.
        '''
        if self.grdmesh['grdsmooth']['activate'] is False:
            self.StrainRateUnSmooth()
        else:
            self.StrainRateSmooth()
    
    
    def StrainRateUnSmooth(self, gridweight=300):
        '''
        Estimate strain rate for each grid point.
        
        Input:
            gridweight = default is 300 km
        '''

        distazim = Grid_Site_DistAizm(self.gdp.llh, self.gpsvelo.llh)        
        maxdist  = self.grdmesh['maxdist']
        minsite  = self.grdmesh['minsite']
        gps      = self.gpsvelo
        print('*************')

        with open(self.strnfile, 'w') as fid:
            fid.write("#  Lon    Lat     Ve     Vn        Exx        Exy        Eyy      omega         E1         E2      shear   dilation  sec_inv_strn  theta\n")

        # each grid point
        for i in range(len(self.gdp.llh)):
            logging.info('{} grid point'.format(i))
            L   = np.ones(7)*np.NaN
            idx = np.where(distazim.dist[i]<maxdist)[0]

            if len(idx)<minsite:
                logging.warning('{:d} GPS sites < {}km around the {:d}th grid point at {:5.2f} {:5.2f}'.format(len(idx),
                                 maxdist, i+1, self.gdp.llh[i,0], self.gdp.llh[i,1]))
                continue
            else:
#               idx  = np.argsort(distazim.dist[i])[0:minsite]
                ve   = gps.ve[idx]
                vn   = gps.vn[idx]
                se   = gps.se[idx]
                sn   = gps.sn[idx]
                gridweight = distazim.dist[i][np.argsort(distazim.dist[i])[minsite-1]]*1.5
                
            if self.grdmesh['chkazim'] is True:
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
                            i+1, self.gdp.llh[i,0], self.gdp.llh[i,1]))
                    continue

            # debug
            if debug == True:
                print('-----------------------------------------------------')
                print('site  distance  azimuth')
                print('grid = {:d} {:8.2f} {:8.2f}'.format(i, self.gdp.llh[i,0], self.gdp.llh[i,1]))
                print('gridweight = {} nsite = {}'.format(gridweight, len(idx)))
                for ii in idx:
                    print('{:s} {:8.2f} {:8.2f}'.format(gps.sitename[ii], distazim.dist[i, ii], distazim.azim[i, ii]))
            
            # GPS sites < maxdist 
            x,y     = np.zeros(len(idx)), np.zeros(len(idx))
            gps_llh = gps.llh[idx]
            
            # Compute local coordinate in east and north
            for j in range(len(idx)):
                x[j], y[j] = llh2utm(gps_llh[j], self.gdp.llh[i])

            # Compute strain rate tensor
            L = self.strainrate(x, y, ve, vn, se, sn, gridweight)

            # Output results to file
            with open(self.strnfile, 'a') as fid:
                fid.write('{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:8.2f}\n'.format(
                        self.gdp.llh[i,0], self.gdp.llh[i,1],
                        L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7],
                        L[8], L[9], L[10], L[11]))
        return
    
    
    def StrainRateSmooth(self):
        '''
        '''
        distazim = Grid_Site_DistAizm(self.gdp.llh, self.gpsvelo.llh)        
        maxdist  = self.grdmesh['maxdist']
        minsite  = self.grdmesh['minsite']
        gps      = self.gpsvelo

        with open(self.strnfile, 'w') as fid:
            fid.write("#  Lon    Lat     Ve     Vn        Exx        Exy        Eyy      omega         E1         E2      shear   dilation  sec_inv_strn  theta\n")

        # each grid point
        for i in range(len(self.gdp.llh)):
            logging.info('{} grid point'.format(i))
            L   = np.ones(7)*np.NaN
            idx = np.where(distazim.dist[i]<maxdist)[0]

            if len(idx)<minsite:
                logging.warning('{:d} GPS sites < {}km around the {:d}th grid point at {:5.2f} {:5.2f}'.format(len(idx),
                                 maxdist, i+1, self.gdp.llh[i,0], self.gdp.llh[i,1]))
                continue
            else:
#               idx  = np.argsort(distazim.dist[i])[0:minsite]
                ve   = gps.ve[idx]
                vn   = gps.vn[idx]
                se   = gps.se[idx]
                sn   = gps.sn[idx]
                gridweight = distazim.dist[i][np.argsort(distazim.dist[i])[minsite-1]]*1.5
                
            if self.grdmesh['chkazim'] is True:
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
                            i+1, self.gdp.llh[i,0], self.gdp.llh[i,1]))
                    continue

            # debug
            if debug == True:
                print('-----------------------------------------------------')
                print('site  distance  azimuth')
                print('grid = {:d} {:8.2f} {:8.2f}'.format(i, self.gdp.llh[i,0], self.gdp.llh[i,1]))
                print('gridweight = {} nsite = {}'.format(gridweight, len(idx)))
                for ii in idx:
                    print('{:s} {:8.2f} {:8.2f}'.format(gps.sitename[ii], distazim.dist[i, ii], distazim.azim[i, ii]))
            
            # GPS sites < maxdist 
            x,y     = np.zeros(len(idx)), np.zeros(len(idx))
            gps_llh = gps.llh[idx]
            
            # Compute local coordinate in east and north
            for j in range(len(idx)):
                x[j], y[j] = llh2utm(gps_llh[j], self.gdp.llh[i])

            # Compute strain rate tensor
            L = self.strainrate(x, y, ve, vn, se, sn, gridweight)



    
class TriStrainRate(StrainRate):
    '''
    TriStranRate is a class representing strain rate estimator using triangular mesh method.
    '''
    
    def __init__(self, cfg):
        '''
        Constructor.
            
        Input:
            cfg = an instance of Config
        '''
        
        super(TriStrainRate, self).__init__(cfg)
        
        self.strnfile = self.trimesh['strainfile']        
        llh           = self.gpsvelo.llh
            
        self.tri      = mtri.Triangulation(llh[:,0], llh[:,1])
        tri_analysis  = mtri.TriAnalyzer(self.tri)
        mask          = tri_analysis.get_flat_tri_mask(0.3)
        self.tri.set_mask(mask)
        logging.info('Total {} triangles, unmask {} triangles'.format(len(self.tri.triangles), len(mask[mask==True])))
        
    
    def StrainRateEst(self):
        '''
        Compute strain rate at the center of each triangular mesh
        '''
        if self.trimesh['trismooth']['activate'] is False:
            self.StrainRateUnSmooth()
        else:
            self.StrainRateSmooth()
    
    
    def StrainRateUnSmooth(self):
        '''
        Estimate strain rate for each triangular patch.

        '''

        gps      = self.gpsvelo
        tri      = self.tri
        unmask   = np.where(tri.mask == False)[0]

        with open(self.strnfile, 'w') as fid:
            fid.write("#  Lon    Lat     Ve     Vn        Exx        Exy        Eyy      omega         E1         E2      shear   dilation  sec_inv_strn  theta\n")
        
        # each grid point
        for i in unmask:
            idx = tri.triangles[i]
            ve  = gps.ve[idx]
            vn  = gps.vn[idx]
            se  = gps.se[idx]
            sn  = gps.sn[idx]
            cp  = [np.mean(tri.x[tri.triangles[i]]), np.mean(tri.y[tri.triangles[i]])]

            # Compute local coordinate in east and north
            x, y= np.zeros(3), np.zeros(3)
            for j, v in enumerate(idx):
                x[j], y[j] = llh2utm(gps.llh[v], cp)

            # Compute strain rate tensor
            L = self.strainrate(x, y, ve, vn, se, sn)

            # Output to file            
            with open(self.strnfile, 'a') as fid:
                fid.write('{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:8.2f}\n'.format(
                        cp[0], cp[1],
                        L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7],
                        L[8], L[9], L[10], L[11]))
    
    
    def StrainRateSmooth(self):
        '''
        '''
        print('Under coding')
        pass    


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
        self.sites = np.genfromtxt(pntsfile, usecols=[2], dtype='str')

        
    
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

        with open(self.strnfile, 'w') as fid:
            fid.write("#  Lon    Lat     Ve     Vn        Exx        Exy        Eyy      omega         E1         E2      shear   dilation  sec_inv_strn  theta\n")

        with open('modeled.gmtvec', 'w') as fid2:
            fid2.write("#  Lon    Lat     Ve     Vn        Se        Sn        Cor      site\n")    

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
            with open('modeled.gmtvec', 'a') as fid2:
                idx = np.where(self.sites[i] == gps.sitename)[0]
                fid2.write('{:6.2f} {:6.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {}\n'.format(
                    self.pnts[i,0], self.pnts[i,1], L[0], L[1], gps.se[idx][0], gps.sn[idx][0], 0.0, self.sites[i]))
    
    def StrainRateSmooth(self):
        '''
        '''
        print('Under coding')
        pass


class GPSTimeSeries(object):
    '''
    GPSTimeSeries is a class representing GPS time series.
    '''
    
    def __init__(self, gpsinfo, tstype, tspath, sepoch, eepoch, sitelist=[]):
        '''
        Constructor.
        
        Input:
            gpsinfo  = gps info file, containing lon, lat, hei, site
            tstype   = type of GPS time series
            tspath   = path of GPS time series
            sepoch   = start epoch in decimal year
            eepoch   = end epoch in decimal year
            sitelist = only data at sitelist are loaded
        '''
        self._LoadGPSInfo(gpsinfo, sitelist=sitelist)
        self._LoadGPSTimeseries(tstype, tspath, sepoch, eepoch)

    def _LoadGPSInfo(self, gpsinfo, sitelist=[]):
        '''
        Load GPS infomation from input file gpsinfo.
        
        Input:
            gpsinfo = gps info file, containing lon, lat, hei, site
        '''
        if os.path.isfile(gpsinfo) == False:
            logging.warning('GPS info file {} does not exit.'.format(gpsinfo))
            sys.exit()
        
        info = np.genfromtxt(gpsinfo, comments='#')
        site = np.genfromtxt(gpsinfo, usecols=[3], dtype='U', comments='#')
        self.gps_llh  = info[:,[0,1]]
        self.sitename = site        

        # Extract user defined sites
        if len(sitelist) > 0:
            idx = [np.where(self.sitename==sitelist[i])[0][0] for i in range(len(sitelist))]
            self.gps_llh  = self.gps_llh[idx]
            self.sitename = site[idx]
        return
    
    def _LoadGPSTimeseries(self, tstype, tspath, sepoch, eepoch):
        '''
        Load GPS time series.
        
        Input:
            tstype   = type of GPS time series
            tspath   = path of GPS time series
            sepoch   = start epoch in decimal year
            eepoch   = end epoch in decimal year
        '''
        tslist = []
        idx    = []
        if tstype == 'pos':
            for i, stname in enumerate(self.sitename):
                if len(glob.glob(os.path.join(tspath, stname+'*.pos'))) == 0:
                    logging.debug('Cannot find posfile for site {}'.format(stname))
                    continue
                posfile = glob.glob(os.path.join(tspath, stname+'*.pos'))[0]
                if os.path.exists(posfile) == False:
                    logging.debug('Cannot find posfile for site {}'.format(stname))
                    continue
                pos     = PosData(posfile)
                if min(pos.decyr)> eepoch or max(pos.decyr)< sepoch:
                    logging.debug('No records between {} and {} at {}'
                                  .format(sepoch, eepoch, stname))
                    continue
                else:
                    tslist.append(PosData(posfile))
                    idx.append(i)
                    logging.info('Posfile {} is loaded.'.format(posfile))
        elif tstype == 'dat':
            for i, stname in enumerate(self.sitename):
                if len(glob.glob(os.path.join(tspath, stname+'*_obs.dat'))) == 0:
                    logging.debug('Cannot find dat file for site {}'.format(stname))
                    continue
                neufile = glob.glob(os.path.join(tspath, stname+'*_obs.dat'))[0]
                ios     = IOSData(neufile)
                if min(ios.decyr)> eepoch or max(ios.decyr)< sepoch:
                    logging.debug('No records between {} and {} at {}'
                                  .format(sepoch, eepoch, stname))
                    continue
                else:
                    tslist.append(IOSData(neufile))
                    idx.append(i)
                    logging.debug('Iosfile {} is loaded.'.format(neufile))
        
        # update gps_llh and site name
        self.gps_llh  = self.gps_llh[idx]
        self.sitename = self.sitename[idx]
        logging.info('Time series from {} stations are loaded'.format(len(tslist)))
        
        if tstype == 'pos':
            self._ResetGPSTimeseries(tslist, sepoch, eepoch)
        elif tstype == 'dat':
            self._AlignGPSTimeseries(tslist, sepoch, eepoch)
        return

    def _ResetGPSTimeseries(self, tslist, sepoch, eepoch, tstype='pos'):
        '''
        Reset GPS time series from longitude and latitude in pos file. The position
        time series are set to relative to the same reference epoch.
        
        Input:
            tslist   = a list of PosData instance
            sepoch   = start epoch in decimal year
            eepoch   = end epoch in decimal year
        
        '''
        
        smjd   = gpstime.decyrs_to_mjd(sepoch)
        emjd   = gpstime.decyrs_to_mjd(eepoch)
        mjd    = np.arange(smjd, emjd)
        decyrs = np.arange(sepoch+0.5/365.25, eepoch+0.5/365.25, 1/365.25)
        ndays  = len(mjd)
        nsite  = len(tslist)
        lat    = np.nan*np.ones((ndays, nsite))
        lon    = np.nan*np.ones((ndays, nsite))
        N      = np.nan*np.ones((ndays, nsite))
        E      = np.nan*np.ones((ndays, nsite))
        SN     = np.nan*np.ones((ndays, nsite))
        SE     = np.nan*np.ones((ndays, nsite))

        # for each site
        for i in range(nsite):
            # for each day
            for j in range(ndays):      
                idx = np.where(np.abs(tslist[i].MJD-mjd[j])<0.015)[0]
                if len(idx)==1:
                    N[j,i]    = tslist[i].N[idx]
                    E[j,i]    = tslist[i].E[idx]
                    SN[j,i]   = tslist[i].SN[idx]
                    SE[j,i]   = tslist[i].SE[idx]
#                   lat[j,i]  = tslist[i].NLat[idx]
#                   lon[j,i]  = tslist[i].ELon[idx]

            # find epoch index for ith site where has data
            idx = np.where(np.isnan(N[:,i])==False)[0]
            if len(idx) == 0:
                continue
            yr0 = decyrs[idx[0]]
            
            
            # compute velocity ve and vn
            ransac = linear_model.RANSACRegressor()
            yr     = np.array(tslist[i].decyr).reshape((len(tslist[i].decyr),1))
            ransac.fit(yr, tslist[i].N)
            vn    = ransac.estimator_.coef_[0]
            ransac.fit(yr, tslist[i].E)
            ve    = ransac.estimator_.coef_[0]
            logging.info("Velocity for {} Ve={:6.2f} mm/yr Vn={:6.2f} mm/yr".format(tslist[i].site, ve, vn))

            for j in idx:
                E[:,i] = E[:,i] - (E[idx[0],i] - (yr0-sepoch)*ve)
                N[:,i] = N[:,i] - (N[idx[0],i] - (yr0-sepoch)*vn)
            
#            for j in range(len(idx)):
            # Fix a bug by Zhao Bin, 2021-05-12
#           for j in idx:
#               e, n   = llh2localxy([lon[j,i], lat[j,i]], [lon[idx[0],i], lat[idx[0],i]])
#               e, n   = e*1e6, n*1e6
#               E[j,i] = e - (yr0-sepoch)*ve
#               N[j,i] = n - (yr0-sepoch)*vn
                
        self.N  = N
        self.E  = E
        self.SN = SN
        self.SE = SE
        
        return

    def _AlignGPSTimeseries(self, tslist, sepoch, eepoch):
        '''
        Align GPS time series to the same reference epoch.
        
        Input:
            tslist   = a list of PosData instance
            sepoch   = start epoch in decimal year
            eepoch   = end epoch in decimal year
        
        '''

        decyrs = np.arange(sepoch+0.5/365.25, eepoch+0.5/365.25, 1/365.25)
        ndays  = len(decyrs)
        nsite  = len(tslist)
        N      = np.nan*np.ones((ndays, nsite))
        E      = N.copy()
        SN     = N.copy()
        SE     = N.copy()

        for i in range(len(tslist)):
            for j in range(ndays):
                # The code is very sensitive to the threshold value, for example 0.0013
                idx = np.where(np.abs(tslist[i].decyr-decyrs[j])<0.0013)[0]
                if len(idx)==1:
                    SN[j,i] = tslist[i].SN[idx]
                    SE[j,i] = tslist[i].SE[idx]
                    N[j,i]  = tslist[i].N[idx]
                    E[j,i]  = tslist[i].E[idx]

            # find epoch index for ith site where has data
            idx = np.where(np.isnan(N[:,i])==False)[0]
            if len(idx) == 0:
                continue
            yr0 = decyrs[idx[0]]
            

            # find epoch index for ith site where has data
            # compute velocity ve and vn
            ransac = linear_model.RANSACRegressor()
            yr     = np.array(tslist[i].decyr).reshape((len(tslist[i].decyr),1))
            ransac.fit(yr, tslist[i].N)
            vn    = ransac.estimator_.coef_[0]
            ransac.fit(yr, tslist[i].E)
            ve    = ransac.estimator_.coef_[0]
            logging.info("Velocity for {} Ve={:6.2f} mm/yr Vn={:6.2f} mm/yr".format(tslist[i].site, ve, vn))
            E[:,i] = E[:,i] - (E[idx[0],i] - (yr0-sepoch)*ve)
            N[:,i] = N[:,i] - (N[idx[0],i] - (yr0-sepoch)*vn)
                
        self.N  = N
        self.E  = E
        self.SN = SN
        self.SE = SE
        np.savetxt('N.txt', N)
        np.savetxt('E.txt', E)
        
        return

class StrainTimeseries(Strain):
    '''
    StrainTimeseries is a class representing strain estimator in time series mode
    '''  
    def __init__(self, cfg, sitelist=[]):
        '''
        Constructor.
        
        Input:
            cfg = an instance of class Config.
        '''
        gpsinfo = cfg['strain_timeseries']['gpsinfo']
        tstype  = cfg['strain_timeseries']['tstype']
        tspath  = cfg['strain_timeseries']['tspath']
        sepoch  = cfg['strain_timeseries']['sepoch']
        eepoch  = cfg['strain_timeseries']['eepoch']
        grdmesh = cfg['strain_timeseries']['grdmesh']

        # save GPS time series at pickle file for reuse.        
        if cfg['strain_timeseries']['gpsts'] == 'SAVE':
            gpsts   = GPSTimeSeries(gpsinfo, tstype, tspath, sepoch, eepoch, sitelist=sitelist)
            with open('gps_ts.pkl', 'wb') as fid:
                import pickle
                data=pickle.dumps(gpsts)
                fid.write(data)
                logging.info('GPS time series are stored in gps_ts.pkl.')
        # load GPS time series from pickle file.
        elif cfg['strain_timeseries']['gpsts'] != "":
            pklfile = cfg['strain_timeseries']['gpsts']
            with open(pklfile, 'rb') as fid:
                import pickle
                gpsts = pickle.loads(fid.read())
                logging.info('GPS time series are loaded from {}'.format(pklfile))
        # Load GPS time series from ASCII files
        else:
            gpsts   = GPSTimeSeries(gpsinfo, tstype, tspath, sepoch, eepoch)

                    
        self.N        = gpsts.N
        self.E        = gpsts.E
        self.SN       = gpsts.SN
        self.SE       = gpsts.SE
        self.gps_llh  = gpsts.gps_llh
        self.sitename = gpsts.sitename
        self.tstype   = tstype
        self.tspath   = tspath
        self.grdmesh  = grdmesh
        self.sepoch   = sepoch
        self.eepoch   = eepoch
                
        return


class GrdStrainTimeseries(StrainTimeseries):
    '''
    GrdStrainTimeseries is a class representing strain time series estimator in grid mesh mode
    '''
    def __init__(self, cfg):
        '''
        Constructor.
        
        Input:
            cfg  = an instance of class Config
        '''
        super(GrdStrainTimeseries, self).__init__(cfg)
        
        grdmesh       = self.grdmesh
        self.gdp      = GridPoint(grdmesh['slon'], grdmesh['elon'],\
                                  grdmesh['slat'], grdmesh['elat'],\
                                  grdmesh['dn']  , grdmesh['de'])
           
    
    def StrainTimeseriesEst(self):
        '''
        Estimate strain timeseries.
        '''

        if self.grdmesh['grdsmooth']['activate'] is False:
            self.StrainTimeseriesUnSmooth()
        else:
            self.StrainTimeseriesSmooth()
        
    def StrainTimeseriesUnSmooth(self, grdweight=300):
        '''
        Estimater strain time series with no Laplican smoothing applied.

        Input:
            grdweight = distance-decaying constant
        '''
        
        # distance and azimuth from grid points to GPS stations
        distazim = Grid_Site_DistAizm(self.gdp.llh, self.gps_llh)        
        maxdist  = self.grdmesh['maxdist']
        minsite  = self.grdmesh['minsite']

        # for each grid point
        for i in range(len(self.gdp.llh)):
            
            # find the stations with a radius maxdist distance
            idx      = np.where(distazim.dist[i]<maxdist)[0]
                      
        
            if len(idx)<minsite:
                logging.warning('There are only {} GPS sites within {} km \
                        around {:5d}th grid point at {:5.2f} {:5.2f}'.format(len(idx),
                                maxdist, i+1, self.gdp.llh[i,0], self.gdp.llh[i,1]))
                continue
            else:
                gridweight = distazim.dist[i][np.argsort(distazim.dist[i])[minsite-1]]*1.5
            
            # check the distribution of stations is around the grid point
            if self.grdmesh['chkazim'] is True:
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
                    logging.warning('The distribution of GPS sites does not reasonable for {}th grid point at {:5.2f} {:5.2f}.'.format(
                            i+1, self.gdp.llh[i,0], self.gdp.llh[i,1]))
                    continue
 
            decyrs = np.arange(self.sepoch, self.eepoch, 1/365.25)
           
            strnfile = 'grd_{:04d}.txt'.format(i)
            with open(strnfile, 'w') as fid:
                fid.write('#{} {}\n'.format(self.gdp.llh[i, 0], self.gdp.llh[i, 1]))
                fid.write('#    decyr      e      n        Exx        Exy        Eyy      omega         E1         E2      shear   dilation  sec_inv_strn  theta\n')

            # for each epoch
            for j, decyr in enumerate(decyrs):

                # for k station
                idx1 = set(idx)
                idx2 = set(np.where(np.isnan(self.N[j,:]) == False)[0])
                idx3 = set(np.where(np.isnan(self.SN[j,:]) == False)[0])
                idx2 = np.array(list(idx1.intersection(idx2, idx3)))

                if len(idx2) < minsite:
                    logging.warning('There are only {} GPS sites within {} km\
 around {:d}th grid point at {:5.2f} {:5.2f}'.format(len(idx2),
                                maxdist, i+1, self.gdp.llh[i,0], self.gdp.llh[i,1]))
                    continue
                x  = distazim.dist[i,idx2]*np.sin(np.deg2rad(distazim.azim[i,idx2]))
                y  = distazim.dist[i,idx2]*np.cos(np.deg2rad(distazim.azim[i,idx2]))
                E  = self.E[j, idx2]
                N  = self.N[j, idx2]
                SE = self.SE[j, idx2]
                SN = self.SN[j, idx2]
                
                L = self.strainrate(x, y, E, N, SE, SN, gridweight)
                
                # Output to file            
                with open(strnfile, 'a') as fid:
                    fid.write('{:10.4f} {:6.2f} {:6.2f} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:8.2f}\n'.format(
                            decyr,
                            L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7],
                            L[8], L[9], L[10], L[11]))
        return
    
    def StrainTimeseriesSmooth(self):
        '''
        '''
        logging.warning('Under coding')
        pass
        
    

class TriStrainTimeseries(StrainTimeseries):
    '''
    TriStrainTimeseries is a class representing strain time series estimator in triangular mesh mode
    '''
    def __init__(self, cfg):
        '''
        Constructor
        '''

        super(TriStrainRate, self).__init__(cfg)
        self.strnfile = self.trimesh['strainfile']        
        llh           = self.gpsvelo.llh
            
        self.tri      = mtri.Triangulation(llh[:,0], llh[:,1])
        tri_analysis  = mtri.TriAnalyzer(self.tri)
        mask          = tri_analysis.get_flat_tri_mask(0.3)
        self.tri.set_mask(mask)
        logging.info('Total {} triangles, unmask {} triangles'.format(len(self.tri.triangles), len(mask[mask==True])))    
    
    

    def StrainTimeseriesEst(self):
        '''
        Estimate strain timeseries.
        '''

        if self.trimesh['grdsmooth']['activate'] is False:
            self.StrainTimeseriesUnSmooth()
        else:
            self.StrainTimeseriesSmooth()
            
    def StrainTimeseriesUnSmooth(self, grdweight=300):
        '''
        Estimater strain time series with no Laplican smoothing applied.
        '''        
        pass
    
    def StrainTimeseriesSmooth(self):
        pass


class UserStrainTimeseries(StrainTimeseries):
    '''
    Calculate strain rate time series at the centroid of user defined GPS stations.
    '''
    def __init__(self, cfg):
        '''
        Constructor.
        '''
        self.stfile = cfg['strain_timeseries']['usrmesh']['sitelist']
        with open(self.stfile, 'r') as fid:
            lines = fid.readlines()
        
        self.sitelists = [line.split() for line in lines]
        alllist  = []
        for ilist in self.sitelists:
            alllist = alllist+ilist
        super(UserStrainTimeseries, self).__init__(cfg, sitelist=alllist)
            

    def StrainTimeseriesEst(self):
        '''
        Estimate strain timeseries.
        '''

        for i in np.arange(len(self.sitelists)):
            print(self.sitelists[i])
            if len(self.sitelists[i]) < 3:
                logging.warning('There are less than 3 sites!')
                continue
            idx_site = []
            for site in self.sitelists[i]:
                idx_site.append(np.where(self.sitename==site)[0][0])
            if len(idx_site) <3:
                logging.info('There are less than 3 sites')
                continue

            cp  = np.mean(self.gps_llh[idx_site], axis=0)
            x,y = np.zeros(len(idx_site)), np.zeros(len(idx_site))
            for j, v in enumerate(idx_site):
                x[j], y[j] = llh2localxy(self.gps_llh[v], cp)
            decyrs = np.arange(self.sepoch, self.eepoch, 1/365.25)

            strnfile = '_'.join(self.sitelists[i])+'.txt'
            with open(strnfile, 'w') as fid:
                fid.write('#{:8.2f} {:8.2f}\n'.format(cp[0], cp[1]))
                fid.write("#    decyr      e      n        Exx        Exy        Eyy      omega         E1         E2      shear   dilation  sec_inv_strn  theta\n")
            # for each epoch
            for j, decyr in enumerate(decyrs):
                if j >= self.E.shape[0]: continue
                E  = self.E[j, idx_site]
                N  = self.N[j, idx_site]
                SE = self.SE[j, idx_site]
                SN = self.SN[j, idx_site]
                ad = np.hstack((E, N, SE, SN))
                idx = np.where(np.isnan(E)==False)[0]
#               E   = E[idx]
#               N   = N[idx]
#               SE  = SE[idx]
#               SN  = SN[idx]
                if sum(np.isnan(E)) != 0:
#                   print(j, decyr)
#                   print(decyr, E, N, SE, SN)
                    continue
                if SE[0]>5 or SE[1]>5 or SE[2]>5 or SN[0]>5 or SN[1]>5 or SN[2]>5:
                    continue
                L = self.strainrate(x, y, E, N, SE, SN)

                # Output to file            
                with open(strnfile, 'a') as fid:
                    fid.write('{:10.5f} {:6.2f} {:6.2f} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:8.2f}\n'.format(
                            decyr,
                            L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7],
                            L[8], L[9], L[10], L[11]))


if __name__ == '__main__':
    cfg = Config('config.yaml').cfg
    if cfg['strain_rate']['activate'] == True:
        logging.info('Strain rate calculation begin')
        if cfg['strain_rate']['grdmesh']['activate'] == True:
            logging.info('Strain rate calculation for grdmesh')
            strainrate = GrdStrainRate(cfg)
            strainrate.StrainRateEst()
        if cfg['strain_rate']['trimesh']['activate'] is True:
            logging.info('Strain rate calculation for trimesh')
            strainrate = TriStrainRate(cfg)
            strainrate.StrainRateEst()
        if cfg['strain_rate']['usrmesh']['activate'] is True:
            logging.info('Strain rate calculation for usrmesh')
            strainrate = UserStrainRate(cfg)
            strainrate.StrainRateEst()
    if cfg['strain_timeseries']['activate'] is  True:
        if cfg['strain_timeseries']['grdmesh']['activate'] is True:
            strain = GrdStrainTimeseries(cfg)
            strain.StrainTimeseriesEst()
        if cfg['strain_timeseries']['trimesh']['activate'] is True:
            strain = TriStrainTimeseries(cfg)
            strain.StrainTimeseriesEst()
        if cfg['strain_timeseries']['usrmesh']['activate'] is True:
            strain = UserStrainTimeseries(cfg)
            strain.StrainTimeseriesEst()
