# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:40:53 2018

@author: chrmo74
"""

import numpy as np
import matplotlib.pyplot as plt

def fixedp(f,x0,tol=10e-5,maxiter=100):
    """ Fixed point algorithm """
    e = 1
    itr = 0
    while(e > tol and itr < maxiter):
        x = f(x0)      # fixed point equation
        e = np.linalg.norm(x0-x) # error at the current step
        x0 = x
        itr = itr + 1
    return x

def comp_1dB_comp(amp_coeffs):
    def steadystate_amp(amp_coeffs, x):
        amp = np.sqrt(np.abs(x))
        nr_orders = len(amp_coeffs)
        output = 0
        for order_id in range(nr_orders):
            poly_coeff = amp_coeffs[order_id]
            output = output + poly_coeff * amp * amp**(order_id * 2)
        return np.abs(output)**2
    onedB = 10**.1
    f2 = lambda x : steadystate_amp(amp_coeffs, x) * onedB
    x_start = 1.
    b = fixedp(f2, x_start)
    onedB_comp_point = b
    return onedB_comp_point

amp_coeffs = np.array([0.999952-0.00981788j, # order 1
                     -0.0618171+0.118845j, # order 3
                     -1.69917-0.464933j, # order 5
                     3.27962+0.829737j, # order 7
                     -1.80821-0.454331j]) # order 9
onedB_comp_point = comp_1dB_comp(amp_coeffs)

nr_powers = 100
rel_powers_dB = np.linspace(-10,0, nr_powers)
powers = 10**(rel_powers_dB/10.) * onedB_comp_point

her_coeffs1 = amp_coeffs[0] + 2 * powers * amp_coeffs[1] + 6 * powers**2 * amp_coeffs[2] + 24 * powers**3 * amp_coeffs[3] + 120 * powers**4 * amp_coeffs[4]

gains_dB = 10*np.log10(np.abs(her_coeffs1)**2)
plt.figure(1)
plt.plot(rel_powers_dB, gains_dB)
plt.xlabel('received power relative to one-dB compression point [dB]')
plt.ylabel('gain |a₁ₘ|² d[dB]')
plt.show()

for power_id in range(nr_powers):
    print('(', rel_powers_dB[power_id], ',' , gains_dB[power_id], ')')