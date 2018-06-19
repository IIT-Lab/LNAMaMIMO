# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:05:40 2018

@author: chrmo74
"""

import numpy as np

nr_users = 101
nr_paths = 101
delay_spread_s = 3e-6

delays = np.random.uniform(0,delay_spread_s,(nr_users, nr_paths))
pathlosses = np.random.rayleigh(1,(nr_users, nr_paths))

xcoord_wavelengths = np.random.uniform(100,5000, (nr_users, nr_paths))
ycoord_wavelengths = np.random.uniform(-5000,5000, (nr_users, nr_paths))

prefix = 'ver1_'

np.savetxt(prefix+'delays_s.dat', delays)
np.savetxt(prefix+'pathlosses.dat', pathlosses)
np.savetxt(prefix+'xcoord_wavelengths.dat', xcoord_wavelengths)
np.savetxt(prefix+'ycoord_wavelengths.dat', ycoord_wavelengths)
