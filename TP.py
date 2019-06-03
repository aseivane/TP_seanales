#from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from scipy.fftpack import fft, fftshift
from scipy import signal

BW_rg = 38e6
T_rg=10e-6
fs = 50e6
time = np.r_[0:10e-6+1/fs:1/fs]
ind = np.r_[0:500+1:1]
frec = ind*BW_rg/2
window = signal.tukey(256) # ventana de Tukey de 256 muestras


def chirp_rg(func_phase):
	return(np.exp(1j*2*np.pi*func_phase))

def tita(t, k1, k2):
	return (k1*t**2 + t*k2)

X = chirp_rg(tita(time, BW_rg/(2*T_rg),-BW_rg/2))

#X=np.sin((2*np.pi/10)*ind)		#seno

A = fft(X,2048)/(len(X)/2.0) 


plt.plot(np.real(X))
#plt.plot(np.imag(X))
plt.figure()

plt.plot(np.angle(X))
plt.figure()

freq=np.linspace(-BW_rg/2,BW_rg/2,len(A))
response = 20 * np.log10(np.abs(fftshift(A/abs(A).max())))
plt.plot(freq,response)
plt.xlabel("Frequency [cycles per sample]")
#plt.plot(log10(ind/10),abs(DFT))	#seno
plt.show()


