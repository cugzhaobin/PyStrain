#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os, sys
from pystrain.PyStrain import GrdStrainRate, TriStrainRate, UserStrainRate, GrdStrainTimeseries, TriStrainTimeseries, UserStrainTimeseries

def main():
    if os.path.isfile('config.yaml') == False:
        logging.fatal('Please prepare config.yaml file!')
        sys.exit()
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

if __name__ == '__main__':
    main()
