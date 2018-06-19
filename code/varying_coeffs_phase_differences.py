# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:49:17 2018

@author: chrmo74
"""

import numpy as np
import matplotlib.pyplot as plt

nr_realisations = 800
nr_antennae = 100
phase_spread_deg = 90
phase_spread_rad = phase_spread_deg / 180. * np.pi
nr_angles = 1000
sine_angles_rad = (np.arange(nr_angles) - nr_angles/2.) / nr_angles * np.pi * 2.


array_gains = np.zeros(nr_angles)
for realisation_id in range(nr_realisations):
    phase_deviations = np.random.uniform(-phase_spread_rad, phase_spread_rad, nr_antennae)
    
    for angle_id in range(nr_angles):
        sine_angle_rad = sine_angles_rad[angle_id]
        array_gains[angle_id] += abs(np.sum(np.exp(1j * (np.arange(nr_antennae) * sine_angle_rad + phase_deviations))))**2 / nr_realisations

print(10 * np.log10(max(array_gains)))

plt.figure()
plt.axhline(0, color='grey')
plt.axhline(20, color='grey')
plt.axhline(40, color='grey')
plt.plot(180. * sine_angles_rad / np.pi, 10 * np.log10(array_gains))
plt.xlabel("difference in sine angle φ [degree]", fontsize=12)
plt.ylabel("average gain E[G(φ)]/|a₃|²", fontsize=12)
plt.ylim((-10, 50))
plt.text(-180, 45,'δ='+str(phase_spread_deg)+'°',
     horizontalalignment='left',
     verticalalignment='top',
     fontsize = 12)
