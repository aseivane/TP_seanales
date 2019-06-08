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
chirp_conj_fft = fft(np.conjugate(chirp),2048)/(len(chirp)/2.0)

A = ifft(A1*chirp_conj_fft)
A=A[len(chirp)-1:] #le saco 500 puntos por la separacion ya que al invertir la chirp deberia quedar en tiempo negativo pero eso el python no lo ve

if(graph_autocor==1):
	plt.figure()
	plt.ylabel("Chirp(t)")
	plt.xlabel("Tiempo")
	plt.title("Chirp")
	plt.plot(chirp)

	#plt.figure()
	plt.ylabel("Chirp2(t)")
	plt.xlabel("Tiempo")
	plt.plot(X1)

	#plt.figure()
	A=abs(A)*400
	plt.plot(A)
	plt.title("Autocorrelacion")
	plt.ylabel("IFFT(A1*A2)")
	plt.xlabel("Tiempo")
	plt.show()






####################################     COMPRESION EN RANGO     ##################################################################


#***** Verificacion de enfoque *****
mat = scipy.io.loadmat('SAR_data_sint.mat')
mat=mat['data_sint']
mat=np.transpose(mat)

#mat = scipy.io.loadmat('SAR_data_sarat.mat')
#mat=mat['data']

matriz_cruz = np.zeros( ( mat.shape[0] , mat.shape[0] ),dtype=complex ) #armamos la matriz A del tamaño que voy a necesitar

for i in range(0,mat.shape[0]-1):
	fila = mat[i,:]
	fila_flip_fft = fft(np.flip(fila),2048)/(len(fila)/2.0)
	matriz_cruz[i,:] = ifft(fila_flip_fft*chirp_conj_fft)*1000# tiene tamaño 4250 x 2048
'''
matriz_cruz =matriz_cruz[:,int(ceil(1.5*len(chirp)))-1:]

matriz_cruz = np.transpose(matriz_cruz)

final = np.zeros( ( matriz_cruz.shape[0] , mat.shape[1] ), dtype=complex ) #armamos la matriz A del tamaño que voy a necesitar


for i in range(0,matriz_cruz.shape[0]-1):
	columna = matriz_cruz[i,:]
	columna_flip_fft = fft(np.flip(columna),2048)/(len(columna)/2.0)
	final[i,:] = ifft(columna_flip_fft*chirp_conj_fft)*1000# tiene tamaño 4250 x 2048

print(fila_flip_fft.shape)
'''
'''
X1 = np.concatenate((mat['data_sint'][2000,:], zeros(len(chirp)-1,dtype=float)), axis=0, out=None)
X1 = np.concatenate((zeros(len(chirp)-1,dtype=float),X1), axis=0, out=None)
A1 = fft(np.flip(X1),2048)/(len(X1)/2.0) 
A= ifft(A1*chirp_conj_fft)*10000
'''
#print(len(A[0,:]))
#fila_2000 = np.concatenate((np.concatenate((zeros(len(chirp)-1,dtype=float),mat['data_sint'][2000,:]), axis=0, out=None), zeros(len(chirp)-1,dtype=float)), axis=0, out=None)
#print(matriz_cruz)

if(graph_foco==1):
	plt.figure()
	#plt.plot(mat['data_sint'][2000,:])
	#plt.plot(chirp)
	matriz_cruz=abs(matriz_cruz)
	#plt.plot(np.abs(matriz_cruz[2000,:]))
	plt.xlabel("Correlacion SAR_data_sint")
	#mat=abs(mat)
	plt.pcolormesh(matriz_cruz)
	plt.colorbar()
	plt.show()