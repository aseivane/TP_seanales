from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import fftshift
import scipy.io

#****** Definimos las constantes y arrays para las funciones ******

win_size=15
BW_rg = 38e6
T_rg=10e-6
fs = 50e6
N=500
time = np.r_[0:10e-6+1/fs:1/fs]
ind = np.r_[0:2*np.pi+(2*np.pi)/500:(2*np.pi)/500]

def chirp_rg(time):
	return(np.exp(1j*2*np.pi*time))


def tita(t, k1, k2):
	return (t*t*k1 + t*k2)

#***** Evaluo la chirp en el array time *****
chirp = chirp_rg(tita(time, BW_rg/(2*T_rg),-BW_rg/2))


#***** Armo el espectrograma para la ventana tukey *****

plt.figure()
window = signal.tukey(win_size) # ventana de Tukey de 256 muestras
f, t, Sxx = signal.spectrogram(chirp, fs, window,  return_onesided=False)
f = np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx, axes=0)
plt.pcolormesh(t,f,Sxx)
plt.title("Espectrograma Tukey")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

#***** Ploteo en tiempo la ventana utilizada *****
plt.figure()
plt.plot(window)
plt.title("Tukey window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.ylim([0, 1.1])

''' no funciona por alguna razon
plt.figure()
A = fft(window,2048)/(len(window)/2.0)  # fft de 2048 puntos
freq = np.linspace(-BW_rg/2, BW_rg/2, len(A))
response = 20 * np.log10(np.abs(fftshift(A/abs(A).max())))
plt.plot(freq,response)
plt.axis([0, T_rg, -120, 0])
plt.title("Frequency response of the Tukey window")
plt.ylabel("Normalized magnitude [dB]")
plt.xlabel("Normalized frequency [cycles per sample]")
'''


#********** Ventana Hanning **********

#***** Armo el espectrograma para la ventana tukey *****

plt.figure()
window = signal.hanning(win_size)
f, t, Sxx = signal.spectrogram(chirp, fs, window,  return_onesided=False)
f = np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx, axes=0)
plt.pcolormesh(Sxx)
plt.title("Espectrograma Hanning")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

#***** Ploteo en tiempo la ventana utilizada *****
plt.figure()
plt.plot(window)
plt.title("Hanning window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.ylim([0, 1.1])
plt.show()
'''
A = fft(window,2048)/(len(window)/2.0)  # fft de 2048 puntos
freq = np.linspace(-BW_rg/2, BW_rg/2, len(A))
response = 20 * np.log10(np.abs(fftshift(A/abs(A).max())))
plt.plot(freq,response)
plt.axis([0, T_rg, -120, 0])
plt.title("Frequency response of the Tukey window")
plt.ylabel("Normalized magnitude [dB]")
plt.xlabel("Normalized frequency [cycles per sample]")'''
