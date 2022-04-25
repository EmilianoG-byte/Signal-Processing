#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
import pywt
import tftb
import scipy.signal as sig


def cwt(name,wavelet,num_bits,data,scales,dt=1,reps=1,image='yes'):
    """    
    
    Computes Continous Wavelet Transform (CWT) coefficients of the input data using different wavelets and plots the
    value of the CWT coefficients as a function of time and scales. 
    
    For a list of all available continous wavelets use: pywt.wavelist(kind='continuous')
    
    scales can be changed based on the input signal. Recommended range: (1,2**n + 1) for n in integers.
    
    input:
    name: (str) name of the channel of data
    wavelet: (str) name of wavelet to use for CWT 
    num_bits: (int) number of bits to 'cut of'from the total data to use in the analysis method
    data: (np.array) array containing the signal. E.g. HDMI, USB.
    scales: (np.array) scales to use for the wavelet transform 
    dt: (float) time step between two consecutive time domain samples, default is 1 second
    reps: (int) number of repetitions over which average will be computed, default is 1 to compute only 1 window without averaging. If
    reps>1, it computes the coefficients over #reps non-overlapping windows of data from the original signal and averages result.
    image: (str) 'yes'to save the image as .jpg and 'no' to not save it, default is 'yes'.
    
    
    output: 
    produces 2D color map where the x-axis corresponds to the time array, the y-axis corresponds to scales and the color
    represents the magnitude. Data is scaled to the range [0,abs(wavelet_coefficients).max()] before mapping to colors.
    coef: 2D array containing the coefficients 
    
    
    Note: to map from scales to frequencies one can use pywt.scale2frequency(wavelet, scales)/dt or freqs/dt
    
    Documentation: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
    
    """
    
    data = data[10:] 
    #ignores 10 bits as these are not real signal data. Unlike other methods we do not average so we only need num_bits points
    
    coef = np.zeros((len(scales),num_bits))

    for i in range(reps):
        
        coef_i, freqs = pywt.cwt(data[(i*num_bits):((i+1)*num_bits)],scales,wavelet) 
        #computes the coefficients and corresponding frequencies of wavelet transform

        
        coef += coef_i

        
    coef = coef/reps

    
    plt.figure(figsize = (12,8))
    c = plt.imshow(coef, origin='lower',aspect='auto', interpolation='none', vmin=0, vmax=abs(coef).max(),extent=(0,num_bits*dt,scales[0],scales[-1]))  # doctest: +SKIP
    plt.colorbar(c)
    plt.ylabel('scales')
    plt.xlabel('time')
    if image=='yes':
        plt.savefig(wavelet+'_'+name+'.png') #saving as png since pdf does not compile properly
    plt.show()
    
    return coef, freqs



def spectro_WVD(data,num_bits,dt=1):
    
    """
    
    Computes the Short Time Fourier Transform (STFT) with its times and frequencies for plotting + times, frequencies 
    and coefficients of wigner-ville distribution (WVD).
    
    input: 
    
    data: (np.array) array containing the signal. E.g. HDMI, USB.
    num_bits: (int) number of bits to 'cut of'from the total data to use in the analysis method
    dt: (float) time step between two consecutive time domain samples, default is 1 second.
    
    output: 
    
    f_stft = frequencies of the short fourier transform corresponding to y-axis in spectogram
    t_tsft = times of the short fourier transform corresponding to x-axis in spectogram
    Zxx = coefficients in short fourier transform
    
    tfr_wvd = coefficients of WVD
    ts = time stamps correspondent to x-axis in WVD plot
    f_wvd = frequencies correspondent to y-axis in WVD plot. 
    
    Note: Here Zxx, f_stft need to be shifted as in 'plotting_WVD()' for better representation.
    Default is to have first half with positive frequencies and second half with negative frequencies
    
    documentation:  
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
    https://tftb.readthedocs.io/en/latest/introduction.html
    
    """

    
    signal = data[10:num_bits+10]
    #ignores 10 bits as these are not real signal data. Unlike other methods we do not average so we only need num_bits points
 
    
    freq_s = 1/dt  # sampling frequency
    ts = np.arange(num_bits) * dt  #timestamps
    
    # Calculating Power of the short time fourier transform (SFTF):
    nperseg = 2**6  # window size of the STFT
    f_stft, t_stft, Zxx = sig.stft(signal, freq_s, nperseg=nperseg, 
                               noverlap=nperseg-1, return_onesided=False)
   
        
    
    # Calculating Wigner-Ville Distribution
    #wvd = tftb.processing.WignerVilleDistribution(signal[:len(signal)//2], timestamps=ts[10:(num_bits+10)//2])
    wvd = tftb.processing.WignerVilleDistribution(signal, timestamps=ts)
    tfr_wvd, t_wvd, f_wvd = wvd.run()
    
    return f_stft, t_stft, Zxx, tfr_wvd, ts, f_wvd


def plotting_WVD(f_stft, t_stft, Zxx, tfr_wvd, ts, f_wvd, name):    
    
    """
    
    Function to use together with spectro_WVD(). 
    
    Input: comes from the output of spectro_WVD(), for explanation on input see the corresponding
    comments of the latter function.
    
    output: 
    1st plot: spectogram with y-axis shifted to show the zero frequency in the middle. x-axis corresponds to time domain
    with the magnitude squared of the Zxx coefficients mapped to colors.

    2nd plot: Wigner-Ville distribution plot with y an x axis as for the 1st plot. Coefficients of WVD mapped to colors.
    
    """
    
    # shifting the frequency axis for better representation
    Zxx = np.fft.fftshift(Zxx, axes=0)
    f_stft = np.fft.fftshift(f_stft)
    
    dt = ts[1] - ts[0] #recalculating time step
    
    f, axx = plt.subplots(2, 1,figsize=(10,10))
    
    df1 = f_stft[1] - f_stft[0]  # frequency step
    im = axx[0].imshow(np.real(Zxx * np.conj(Zxx)), aspect='auto', #here Zxx*np.conj(Zxx) = |Zxx|^2 for spectogram
              interpolation=None, origin='lower',
              extent=(ts[0] - dt/2, ts[-1] + dt/2,
    
                      f_stft[0] - df1/2, f_stft[-1] + df1/2))
    axx[0].set_ylabel('frequency [Hz]')
    plt.colorbar(im, ax=axx[0])
    axx[0].set_title('spectrogram')

    # Due to implementation of WT, the maximum frequency is half of
    # the sampling Nyquist frequency, e.g. 125 Hz instead of 250 Hz, and the sampling # is 2 * dt instead of dt
    
    f_wvd = np.fft.fftshift(np.fft.fftfreq(tfr_wvd.shape[0], d= 2* dt))
    df_wvd = f_wvd[1]-f_wvd[0]  # the frequency step in the WVT
    im = axx[1].imshow(np.fft.fftshift(tfr_wvd, axes=0),aspect='auto', origin='lower',
           extent=(ts[0] - dt/2, ts[-1] + dt/2,
                   f_wvd[0]-df_wvd/2, f_wvd[-1]+df_wvd/2),vmin=0) 
    #here we have set the vmin = 0 to see the figures better but in reality the WVD has negative components since data is real
    #vmin=0,vmax=abs(tfr_wvd).max()
    
    axx[1].set_xlabel('time [s]')
    axx[1].set_ylabel('frequency [Hz]')
    plt.colorbar(im, ax=axx[1])
    axx[1].set_title('Wigner-Ville Transform-'+name)
    plt.show()
    
    return


def period(num_bits,data,reps,dt=1):
    
    """
    
    input:
    
    data: (np.array) array containing the signal
    num_bits: (int) number of bits to 'cut of'from the total data to use in the analysis.
    reps: (int) number of repetitions over which average will be computed
    dt: (float) time step between two consecutive time domain samples, default is 1 second.
    
    
    output: 
    f: frequencies associated to time domain signal
    Pxx: Power Spectral Density corresponding to the frequencies from 'f'.
    
    """
    data = data[10:] #10 first bits ignored as these are not real signal data.
    fs = 1/dt #sampling frequency
    f, Pxx_den = signal.periodogram(data[:num_bits], fs) #used to obtain the length of Pxx_den
    Pxx_tot = np.zeros(len(Pxx_den)) #array with equal shape as Pxx_den.
    
    for i in range(reps):
        Pxx_tot += signal.periodogram(data[(i*num_bits):((i+1)*num_bits)],fs)[1] #adding up all the realizations of Pxx_den
    
    Pxx = Pxx_tot/reps #averages over all realizations
    
    return f, Pxx


def main():
    print('You are using this script as your main')
    
if __name__=='__main__':
    main()
