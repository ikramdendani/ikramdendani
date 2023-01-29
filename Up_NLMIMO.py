 # -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:54:31 2021

@author: MAHA
Uplink Massive MIMO - with  PA (Ref1)
"""

#In[1]
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from commpy.modulation import QAMModem
from commpy.utilities import bitarray2dec
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
import decimal
from decimal import getcontext
#import import_ipynb

getcontext().Emax = 600000000000

def hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B):
    a0 = abs(vin)
    theta = np.angle(vin)
    Am = (G * a0) / ((1 + (G * a0 / Vsat) ** (2 * p)) ** (1 / (2 * p)))
    Bm = (A * (a0 ** q)) / ((1 + (a0 / B) ** (q)))
    vout = Am * np.exp(1j * (theta + Bm))
    return vout
def find_K0_sigma2_d(IBO):
    xin = (1 / np.sqrt(2)) * (np.random.randn(1, 1000) + 1j * np.random.randn(1, 1000)) 
    coeff_IBO_m1dB = (
        val_IBO_m1dB * np.sqrt((1 / np.var(xin))) * np.sqrt(10 ** (-IBO / 10))
    )
    vin = coeff_IBO_m1dB * xin
    vout = hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B)    
    K0 = np.mean(vout * np.conj(vin)) / np.mean(np.absolute(vin) ** 2)
    sigma2_d = np.var(vout - K0 * vin)
    return (K0, sigma2_d)

def frange(x,y,jump):
    while x<y:
        yield x
        x+=jump


Mr=10
Mt = 100
M = 16
Nreal = 10000

IBO =5

p = 1.1
q = 4
Vsat = 1.9

G = 16 
A = -345
B = 0.17

val_IBO_m1dB = 0.1185
K0, sigma_2d = find_K0_sigma2_d(IBO)

N_bits = Mr* np.log2(M)

SNRdB_range = list(frange(0,20,2))
BERm = [None]*len(SNRdB_range)
SERm = [None]*len(SNRdB_range)
MUIm = [None]*len(SNRdB_range)
for l in range(0,len(SNRdB_range)):
    SNR = 10**(SNRdB_range[l]/10)
    MUIi = np.zeros((Nreal, 1))
    BERi = np.zeros((Nreal, 1))
    SERi = np.zeros((Nreal, 1))
    SYM = []
    SYMchp = []
    for j in range(Nreal):
       
        H = np.sqrt(1/(2*Mt)) * (np.random.randn(Mt, Mr) + 1j * np.random.randn(Mt, Mr))
        

        bits=np.random.randint(2,size=int(N_bits))
        bitsp = np.reshape(bits,(4,Mr))
        bitsp = np.array(bitsp)
        symbols = bitarray2dec(bitsp)
        
        QAM16 = QAMModem(M)  
        z = QAM16.modulate(bits)
        s = np.reshape(z, (1, Mr)).T
        
         # PA
        coeff_IBO_m1dB=val_IBO_m1dB*np.sqrt((1/np.var(s)))*np.sqrt(10**(-IBO/10));#normalization
        vin = coeff_IBO_m1dB*s
        vout = hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B)
        
        r1 =(np.conj(K0)/abs(K0)**2)*vout/(coeff_IBO_m1dB)
        
        rec = H.dot(r1)
        sigman=np.mean((np.abs(r1) ** 2))/(2*SNR)
        noise1 = np.sqrt(sigman*Mr/Mt)*(np.random.randn(Mt,1) +1j*(np.random.randn(Mt,1)))        
        rec1 = rec + noise1 
        
        H_H=H.conj().T
        P=np.dot(H,H_H)
        #invH=np.linalg.inv(sigman*np.eye(Mt) + P)
        invH=np.linalg.inv(0.008*np.eye(Mt) + P)
        Pre=np.dot(H_H,invH)
        
        r = Pre.dot(rec1)
        #r = np.dot(Pre,rec1)
     
        
        bitchp = QAM16.demodulate(r.T,'hard')
        bitschpp = np.reshape(bitchp,(4,Mr))
        bitschpp = np.array(bitschpp)
        symbolschp = bitarray2dec(bitschpp)
        MUIi[j] = np.mean((np.abs(r - s) ** 2))/np.mean((np.abs(s) ** 2))
        BERi[j] = np.sum(np.abs(bitchp-bits))/N_bits
        SERi[j] = np.sum(symbolschp!=symbols)/(Mr)
        
        
    MUIm[l] = np.mean(MUIi)
    BERm[l] = np.mean(BERi)
    SERm[l] = np.sum(SERi)/Nreal
    print(SERm[l])

plt.figure()
plt.plot(SNRdB_range,SERm,'b-.')
plt.yscale('log')
plt.xlabel('SNR [dB]')
plt.ylabel('SER')      


