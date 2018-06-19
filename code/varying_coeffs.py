# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:43:31 2017

@author: chrmo74
"""


import numpy as np
import matplotlib.pyplot as plt

nr_antennae = 100
mean = -0.0427976467+1j*2.95963e-03
eta = .1
variance = eta*abs(mean)**2

nr_realisations = 10000

nr_phases = 112
phases = np.arange(nr_phases + 1) / float(nr_phases) * 2*np.pi - np.pi
nr_phases += 1

array_gain = np.zeros((nr_phases, nr_realisations))

for realisation_id in range(nr_realisations):
    coeffs = np.sqrt(1 - eta) * mean + np.random.normal(0, np.sqrt(variance/2.), nr_antennae) + 1j * np.random.normal(0, np.sqrt(variance / 2.), nr_antennae)
    for phase_id in range(nr_phases):
        phase = phases[phase_id]
        gain = 0. + 0.j
        for antenna_id in range(nr_antennae):
            gain += coeffs[antenna_id] * np.exp(1j * antenna_id * phase)
        array_gain[phase_id, realisation_id] = abs(gain)**2
    
#%%

# spatial pattern:
mean_array_gain = np.mean(array_gain, axis=1) / abs(mean)**2

for phase_id in range(nr_phases):
    print(str(phases[phase_id])+" " + str(mean_array_gain[phase_id]))

plt.figure(1)
plt.clf()
plt.plot(phases/np.pi*180, 10 * np.log10(np.real(mean_array_gain)))
plt.ylim(0,50)
plt.xlabel("difference in sine angles [deg]")
plt.ylabel("array gain [dB]")
plt.show()

# two users:    
nrAntennae = 100
nrAngles = 600

sineAngles = (np.arange(nrAngles) - nrAngles/2.) / float(nrAngles) * np.pi * 2.

user1SineAngle = np.pi / 2. * .1
user2SineAngle = -np.pi / 2. * .5

angleDiff = user1SineAngle - user2SineAngle

summa1 = np.zeros(nrAngles, dtype=complex)
summa2 = np.zeros(nrAngles, dtype=complex)
for antennaID in range(nrAntennae):
    summa1 += np.exp(-1j * sineAngles * antennaID) * np.cos(angleDiff * antennaID)
    summa2 += np.exp(-1j * sineAngles * antennaID) * np.sin(angleDiff * antennaID)

arrayGains = abs(summa1)**2 + abs(summa2)**2

for angleID in range(int(nrAngles / 2)):
    print(str(sineAngles[angleID]) + ' ' + str(arrayGains[angleID]))
    
print(angleDiff)

plt.figure(2)
plt.clf()
plt.plot(sineAngles, 10 * np.log10(arrayGains))
plt.plot(angleDiff, 0, 'x', color='red')
plt.ylim((-100, 40))
plt.xlabel("sine angles")
plt.ylabel("array gain [dB]")
plt.show()
