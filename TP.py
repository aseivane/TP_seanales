#from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from scipy.fftpack import fft, fftshift, ifft
from scipy import signal
import scipy.io

#****** Definimos las constantes y arrays para las funciones******
graph_chirp = 0
graph_autocor = 0
graph_foco = 1


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

#***** Evaluo la chirp en el array time *****
chirp = chirp_rg(tita(time, BW_rg/(2*T_rg),-BW_rg/2))
chirp_fft = fft(chirp,2048)/(len(chirp)/2.0) 

if(graph_chirp==1):
	plt.plot(np.real(chirp))
	#plt.plot(np.imag(X))
	plt.ylabel("Chirp(t)")
	plt.xlabel("Tiempo")
	plt.figure()

	plt.plot(np.angle(chirp))
	plt.ylabel("Fase")
	plt.xlabel("Tiempo")
	plt.figure()

	freq=np.linspace(-BW_rg/2,BW_rg/2,len(chirp_fft))
	response = 20 * np.log10(np.abs(fftshift(chirp_fft/abs(chirp_fft).max())))
	plt.plot(freq,response)
	plt.xlabel("Frequency [cycles per sample]")
	plt.show()


#***** Le agrego ceros a la chirp a los costados para ver que cuando haga la autocorrelacion me quede la delta en el comienzo ******
X1 = np.concatenate((chirp, zeros(len(chirp)-1,dtype=float)), axis=0, out=None)
X1 = np.concatenate((zeros(len(chirp)-1,dtype=float),X1), axis=0, out=None)

A1 = fft(np.flip(X1),2048)/(len(X1)/2.0) 
A2 = fft(np.conjugate(chirp),2048)/(len(chirp)/2.0)

A = ifft(A1*A2)
A=A[500:] #le saco 500 puntos por la separacion ya que al invertir la chirp deberia quedar en tiempo negativo pero eso el python no lo ve

if(graph_autocor==1):
	plt.figure()
	plt.ylabel("Chirp(t)")
	plt.xlabel("Tiempo")
	plt.title("Chirp")
	plt.plot(chirp)

	plt.figure()
	plt.ylabel("Chirp2(t)")
	plt.xlabel("Tiempo")
	plt.plot(X1)

	plt.figure()
	A=abs(A)
	plt.plot(A)
	plt.title("Autocorrelacion")
	plt.ylabel("IFFT(A1*A2)")
	plt.xlabel("Tiempo")
	plt.show()






####################################     COMPRESION EN RANGO     ##################################################################


#***** Verificacion de enfoque *****
mat = scipy.io.loadmat('SAR_data_sint.mat')
'''
A = np.zeros( ( len(mat['data_sint'][:,0]), len(mat['data_sint'][0,:]) ), dtype=complex )/2.0 #armamos la matriz A del tamaño que voy a necesitar

for i in range(0,len(mat['data_sint'][:,0])-1):
	
	X1 = np.concatenate((mat['data_sint'][i,:], zeros(len(chirp)-1,dtype=float)), axis=0, out=None)
	X1 = np.concatenate((zeros(len(chirp)-1,dtype=float),mat['data_sint'][i,:]), axis=0, out=None)
	A1 = fft(np.flip(X1),2048)/(len(X1)/2.0) 
	#print(type(ifft(A1*A2)))
	A[i] = ifft(A1*A2)
	#A[i]=A[i][500:]
	break
'''

X1 = np.concatenate((mat['data_sint'][2000,:], zeros(len(chirp)-1,dtype=float)), axis=0, out=None)
X1 = np.concatenate((zeros(len(chirp)-1,dtype=float),mat['data_sint'][2000,:]), axis=0, out=None)
A1 = fft(np.flip(X1),2048)/(len(X1)/2.0) 
A= ifft(A1*A2)

if(graph_foco==1):
	plt.figure()
	A=abs(A)
	#plt.pcolormesh(A)
	plt.plot(A)
	plt.xlabel("Correlacion SAR_data_sint")
	plt.show()
