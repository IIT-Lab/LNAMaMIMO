# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:45:36 2018

@author: Christopher Mollén
"""

import numpy as np
import matplotlib.pyplot as plt


def rrcos(nr_samples, oversamp_factor, rolloff):
    start_time = -1 * nr_samples / float(oversamp_factor) / 2.
    stop_time = -1 * start_time
    sample_instants = np.linspace(start_time, stop_time, nr_samples)
    denominator = np.pi * sample_instants * (1 - (4.*rolloff*sample_instants)**2)
    time_samples = (np.sin(np.pi*sample_instants*(1-rolloff)) + 4.*rolloff*sample_instants*np.cos(np.pi*sample_instants*(1+rolloff))) / denominator
    time_samples[sample_instants == 0] = 1. + rolloff * (4./np.pi - 1.)
    
    return np.fft.ifftshift(time_samples)

def pam(symbols, pulse, oversamp_factor):
    nr_symbols = len(symbols)
    signal = np.zeros(nr_symbols * oversamp_factor, dtype = complex)
    signal[::oversamp_factor] = symbols
    output = freq_conv(signal,pulse)
    return output

def pa(insig, poly_coeffs):
    output = (poly_coeffs[0] * insig
        + poly_coeffs[1] * insig*abs(insig)**2
        + poly_coeffs[2] * insig*abs(insig)**4
        + poly_coeffs[3] * insig*abs(insig)**6
        + poly_coeffs[4] * insig*abs(insig)**8)
    return output

def freq_conv(sig1, sig2):
    nr_out_samps = len(sig1) + len(sig2) - 1
    first_neg_id = int(np.ceil(len(sig1)/2.) + 1)
    sig1_zp = np.zeros(nr_out_samps, dtype=complex)
    sig1_zp[:first_neg_id] = sig1[:first_neg_id]  
    sig1_zp[-len(sig1)+first_neg_id:] = sig1[first_neg_id:]
    first_neg_id = int(np.ceil(len(sig2)/2.) + 1)
    sig2_zp = np.zeros(nr_out_samps, dtype=complex)
    sig2_zp[:first_neg_id] = sig2[:first_neg_id]
    sig2_zp[-len(sig2)+first_neg_id:] = sig2[first_neg_id:]
    sig1_freq = np.fft.fft(sig1_zp)
    sig2_freq = np.fft.fft(sig2_zp)
    return np.fft.ifft(sig1_freq * sig2_freq)    

def compute_NMSE(est, symbol):
    correlation = np.mean(est*np.conj(symbol))
    symbol_power = np.mean(abs(symbol)**2)
    est_power = np.mean(abs(est)**2)
    NMSE = 1 - abs(correlation)**2/symbol_power/est_power
    return NMSE, correlation

def channel_out(tx_signal, delays_s, pathlosses, carrier_freq_Hz, scatterer_poss_wl, nr_antennae):
    nr_samples = len(tx_signal)
    rel_freqs_Hz = (np.arange(nr_samples) - np.ceil(nr_samples/2.)) / nr_samples * oversamp_factor * baudrate_Hz
    freqs_Hz = rel_freqs_Hz + carrier_freq_Hz
    freqs_Hz = np.fft.ifftshift(freqs_Hz)
    nr_paths = len(delays_s)
    tx_signal_freq = np.fft.fft(tx_signal)
    rx_signals = np.zeros((nr_antennae,nr_samples),dtype=complex)
    for antenna_id in range(nr_antennae):
        channel_freq = np.zeros(nr_samples, dtype=complex)
        for path_id in range(nr_paths):
            pathloss = pathlosses[path_id]
            delay_s = delays_s[path_id]
            scatterer_pos_wl = scatterer_poss_wl[path_id]
            path_difference = abs(scatterer_pos_wl) - abs(scatterer_pos_wl - .5j*antenna_id)
            channel_freq += pathloss * np.exp(-2j*np.pi*freqs_Hz*delay_s) * np.exp(-2j * np.pi * path_difference * freqs_Hz / carrier_freq_Hz)
        rx_signal_freq = tx_signal_freq * channel_freq
        rx_signals[antenna_id,:] = np.fft.ifft(rx_signal_freq)#tx_signal#
    return rx_signals

def demodulate(signal, pulse, oversamp_factor, nr_symbols):
    matched_output = freq_conv(signal, pulse)
    samp_instants = (np.arange(nr_symbols) - int(np.ceil(nr_symbols/2.))+1) * oversamp_factor
    
    pulse_energy = sum(abs(pulse)**2)
    symbols = matched_output[samp_instants]/pulse_energy
    symbols = np.fft.ifftshift(symbols)
    return symbols

def randomQAM(nr_symbols):
    symbol_nr = np.random.randint(16, size=nr_symbols)
    constellation={0 : -3-3j,
        1 : -3-1j,
        2 : -3+3j,
        3 : -3+1j,
        4 : -1-3j,
        5 : -1-1j,
        6 : -1+3j,
        7 : -1+1j,
        8 :  3-3j,
        9 :  3-1j,
        10 :  3+3j,
        11 :  3+1j,
        12 :  1-3j,
        13 :  1-1j,
        14 :  1+3j,
        15 :  1+1j,
    }
    symbols = np.zeros(nr_symbols, dtype=complex)
    for symbol_id in range(nr_symbols):
        symbols[symbol_id] = constellation[symbol_nr[symbol_id]]
    energy = (16-1)*2**2/6.
    energy=2/3.*(16-1)
    symbols = symbols / np.sqrt(energy)
    return symbols
    
def add_CP(symbols, CP_length):
    if CP_length==0:
        return symbols
    else:
        return np.concatenate((symbols[-CP_length:],symbols))

def zero_padded_fft(sig,nr_samples):
    nr_input_samples = len(sig)
    most_neg_id = int(np.ceil(len(sig)/2.) + 1)
    zp_sig = np.zeros(nr_samples, dtype=complex)
    zp_sig[:most_neg_id] = sig[:most_neg_id]  
    zp_sig[-nr_input_samples+most_neg_id:] = sig[most_neg_id:]
    return np.fft.fft(zp_sig)

def load_parameters(nr_users, nr_paths, parameter):
    if nr_users > 100 and nr_paths > 100:
        raise ValueError('Too many users or too many paths.')
    else:
        if parameter == 'pathlosses':
            data = np.loadtxt('set_pathlosses.dat')
        elif parameter == 'coordinates':
            xcoord = np.loadtxt('set_xcoord_wavelengths.dat')
            ycoord = np.loadtxt('set_ycoord_wavelengths.dat')
            data = xcoord + 1j * ycoord
        elif parameter == 'delays':
            data = np.loadtxt('set_delays_s.dat')
        else:
            raise ValueError('No such data as' + str(parameter))
    if nr_users == -1:
        return data[nr_users, :nr_paths]
    else:
        return data[:nr_users, :nr_paths]

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
        amp = np.sqrt(abs(x))
        nr_orders = len(amp_coeffs)
        output = 0
        for order_id in range(nr_orders):
            poly_coeff = np.sum(amp_coeffs[order_id])
            output = output + poly_coeff * amp * amp**(order_id * 2)
        return np.abs(output)**2
    onedB = 10**.1
    f1 = lambda x : 2*x-steadystate_amp(amp_coeffs, x) * onedB
    f2 = lambda x : steadystate_amp(amp_coeffs, x) * onedB
    x_start = 1.
    a = fixedp(f1, x_start)
    b = fixedp(f2, x_start)
    onedB_comp_point = b
    return onedB_comp_point

np.random.seed(1)

nr_antennae = 100
nr_users = 1
nr_symbols = 30000
rolloff=.22
baudrate_Hz = 20e6
carrier_freq_Hz = 2e9
blocker_carrier_freq = 1 + rolloff
blocker_carrier_freq_Hz = carrier_freq_Hz + blocker_carrier_freq * baudrate_Hz
nr_paths = 1
nr_channel_realisations = 1
delay_spread_s = 3e-6
CP_length = int(np.ceil(delay_spread_s * baudrate_Hz)) + 100
oversamp_factor = 5
rel_blocker_power_dB = 50
rel_blocker_power = 10**(rel_blocker_power_dB/10.)
backoff_dB = -8
backoff = 10**(backoff_dB/10.)
poly_coeffs = [0.999952-0.00981788j, -0.0618171+0.118845j, -1.69917-0.464933j, 3.27962+0.829737j, -1.80821-0.454331j]
onedB_comp_point = comp_1dB_comp(poly_coeffs)

nr_pulse_samples = oversamp_factor * 100 - 1
pulse = rrcos(nr_pulse_samples, oversamp_factor, rolloff)

nr_samples = nr_pulse_samples + ((nr_symbols + CP_length) * oversamp_factor) - 1
nr_useful_symbols = int(nr_symbols / 2.)

reference_symbols = np.zeros(nr_useful_symbols * nr_channel_realisations, dtype=complex)
reference_rx_symbols = np.zeros(nr_useful_symbols * nr_channel_realisations, dtype=complex)

delays_s = load_parameters(nr_users, nr_paths, 'delays')
scatterer_pos_wl = load_parameters(nr_users, nr_paths, 'coordinates')
pathlosses = load_parameters(nr_users, nr_paths, 'pathlosses')/np.sqrt(nr_paths)

delays_blocker_s = load_parameters(-1, nr_paths, 'delays')
scatterer_pos_blocker_wl = load_parameters(-1, nr_paths, 'coordinates')
pathlosses_blocker = load_parameters(-1, nr_paths, 'pathlosses')/np.sqrt(nr_paths)

for channel_realisation_id in range(nr_channel_realisations):
    
    extra_delays_s = np.random.uniform(0,1./carrier_freq_Hz,(nr_users,nr_paths))
    extra_delays_blocker_s = np.random.uniform(0,1./carrier_freq_Hz,nr_paths)
    
    symbols_blocker = np.random.normal(0.,np.sqrt(.5), nr_symbols) + 1j*np.random.normal(0.,np.sqrt(.5), nr_symbols)
    symbols_blocker = add_CP(symbols_blocker, CP_length)
    tx_signal_blocker = np.sqrt(rel_blocker_power)*pam(symbols_blocker, pulse, oversamp_factor)
    tx_signal_blocker = tx_signal_blocker * np.exp(2j*np.pi*blocker_carrier_freq*np.arange(nr_samples)/oversamp_factor)
    rx_signals = channel_out(tx_signal_blocker, delays_blocker_s+extra_delays_blocker_s, pathlosses_blocker, carrier_freq_Hz, scatterer_pos_blocker_wl, nr_antennae)
    
    symbols = np.zeros((nr_users,nr_symbols),dtype=complex)
    tx_signals = np.zeros((nr_users,nr_samples), dtype=complex)
    for user_id in range(nr_users):
        QAM = True
        if QAM:
            symbols[user_id] = randomQAM(nr_symbols)
        else:
            symbols[user_id] = np.random.normal(0.,np.sqrt(.5), (nr_symbols)) + 1j*np.random.normal(0.,np.sqrt(.5), (nr_symbols))
        symbols_with_CP = add_CP(symbols[user_id], CP_length)
        tx_signals[user_id,:] = pam(symbols_with_CP, pulse, oversamp_factor)
        total_delay = delays_s[user_id,:] + extra_delays_s[user_id,:]
        rx_signals += channel_out(tx_signals[user_id,:], total_delay, pathlosses[user_id,:], carrier_freq_Hz, scatterer_pos_wl[user_id,:], nr_antennae)
#   uncomment and comment out next line to normalize received power instantaneously or not
#    rx_signals = rx_signals * np.sqrt(backoff*onedB_comp_point/(rel_blocker_power + nr_users))
    rx_signals = rx_signals * np.sqrt(backoff*onedB_comp_point / np.mean(abs(rx_signals)**2))
    
    amp_rx_signals = np.zeros(np.shape(rx_signals), dtype=complex)
    disc_rx_signals = np.zeros((nr_antennae, nr_symbols), dtype=complex)
    for antenna_id in range(nr_antennae):
        amp_rx_signals[antenna_id,:] = pa(rx_signals[antenna_id,:], poly_coeffs)
        rx_signal = demodulate(amp_rx_signals[antenna_id,:], pulse, oversamp_factor, nr_symbols + CP_length)
        disc_rx_signals[antenna_id,:] = rx_signal[CP_length:]
    
    nr_channel_taps = int(np.ceil(delay_spread_s * baudrate_Hz)) + 50
    disc_channel = np.zeros((nr_antennae, nr_users, nr_channel_taps), dtype=complex)
    for user_id in range(nr_users):
        agg_pulse = freq_conv(pulse,pulse)
        total_delay = delays_s[user_id,:] + extra_delays_s[user_id,:]
        temp_channel = channel_out(agg_pulse, total_delay, pathlosses[user_id,:], carrier_freq_Hz, scatterer_pos_wl[user_id,:], nr_antennae) 
        sample_instants = (np.arange(nr_channel_taps)-int(np.ceil(nr_channel_taps/2.))+1) * oversamp_factor
        sample_instants = np.fft.ifftshift(sample_instants)
        disc_channel[:,user_id,:] = temp_channel[:,sample_instants]
    
    disc_channel_freq = np.zeros((nr_antennae,nr_users,nr_symbols),dtype=complex)
    for antenna_id in range(nr_antennae):
        for user_id in range(nr_users):
            disc_channel_freq[antenna_id,user_id,:] = zero_padded_fft(disc_channel[antenna_id,user_id,:],nr_symbols)
    
    decoder = np.zeros((nr_users, nr_antennae,nr_symbols), dtype=complex)
    rx_symbols_freq = np.zeros((nr_users, nr_symbols), dtype=complex)
    disc_rx_signals_freq = np.fft.fft(disc_rx_signals, axis=1)
    for freq_id in range(nr_symbols):
        decoder = np.linalg.pinv(disc_channel_freq[:,:,freq_id])#np.conj(np.transpose(disc_channel_freq[:,:,freq_id]))#
        rx_symbols_freq[:,freq_id] = decoder.dot(disc_rx_signals_freq[:, freq_id])
    
    rx_symbols = np.fft.ifft(rx_symbols_freq, axis=1)

    reference_symbols[channel_realisation_id * nr_useful_symbols:(channel_realisation_id + 1)*nr_useful_symbols] = symbols[0,nr_useful_symbols:]
    reference_rx_symbols[channel_realisation_id * nr_useful_symbols:(channel_realisation_id + 1)*nr_useful_symbols] = rx_symbols[0,nr_useful_symbols:]

NMSE, correlation = compute_NMSE(reference_rx_symbols, reference_symbols)

print("nr antennae = ", nr_antennae)
print("NMSE =", 10*np.log10(NMSE), "dB")
print("rate =", np.log2(1/NMSE), "bpcu")

plt.figure(1)
user_id = 0
NMSE, correlation = compute_NMSE(rx_symbols[user_id,nr_useful_symbols:], symbols[user_id,nr_useful_symbols:])
symbols_to_plot = rx_symbols[user_id,nr_useful_symbols:] / correlation
plt.scatter(np.real(symbols_to_plot), np.imag(symbols_to_plot), marker='x')
plt.scatter(np.real(symbols[user_id,nr_useful_symbols:]), np.imag(symbols[user_id,nr_useful_symbols:]), marker='o',color='red')
plt.xlabel('in-phase')
plt.ylabel('quadrature-phase')
plt.show()

plt.figure(2)
plt.stem(np.abs(rx_symbols[user_id,nr_useful_symbols:]))
plt.show()

plt.figure(3)
plt.clf()
antenna_id = 0
plt.stem(np.abs(disc_channel[antenna_id, user_id, :]))
plt.xlabel('tap index')
plt.ylabel('modulus of discrete channel tap between antenna ' + str(antenna_id) + ' and user ' + str(user_id))
plt.show()

plt.figure(4)
plt.clf()
plt.plot(np.linspace(0,1,nr_symbols), np.abs(disc_channel_freq[0,0,:]), color="black")
plt.xlabel('normalized frequency')
plt.ylabel('modulus of channel spectrum')
plt.show()
