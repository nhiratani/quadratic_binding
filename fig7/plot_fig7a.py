#
# Readout of unbinding performance
#

from math import *
#import sys
import numpy as np
#from numpy import random as nrnd
#from numpy import linalg as nlg
from scipy import stats as scist
from scipy import special as scisp
from scipy import integrate as integrate

import matplotlib.pyplot as plt
#from pylab import cm

clrs = ['C1', 'C0', 'C3'] #(ext-Oct, Tensor-HRR, Random-quad)
alphas = [1.0, 0.6, 0.6]

Ns = [96]#, 96]#, 192]#, 288]#[80, 160, 320]#64
d = 1
Ls = range(2,21,1)
D = 5000
Tmax = 10000#0#10000

Nlen = len(Ns)
Llen = len(Ls)

L_theos = range(1,41,1)

svfg = plt.figure()
for Nidx in range(Nlen):
    N = Ns[Nidx]
    aerrs = np.zeros((3, Llen))
    for Lidx in range(Llen):
        L = Ls[Lidx]
        festr = 'data/qbind_dict_ext_sKcomp_tensor_HRR_random_error_N' + str(N) + '_d' + str(d) + '_L' + str(L) + '_D' + str(D) + '_Tm' + str(Tmax) + '.txt'
        
        lidx = 0
        for line in open(festr, 'r'):
            ltmp = line[:-1].split(" ")
            aerrs[lidx, Lidx] = float(ltmp[1])
            lidx += 1

    K = 8
    #not clear if sig2_3 = N is more accurate description of the system
    #dsig2s = np.array([[2.0*N/K, 2.0*N/K, 3*N],
    #                   [N, N, 3*N],
    #                   [N, N, 3*N]])
    dsig2s = np.array([[2.0*N/K, 2.0*N/K, N],
                       [N, N, N],
                       [N, N, N]])

    errs_theor = np.zeros((3, len(L_theos)))
    for q in range(3):
        for Lidx in range(len(L_theos)):
            L = L_theos[Lidx]
            sig2s = np.zeros((3,3))
            sig2s[q,0] = L*N + dsig2s[q,0] + dsig2s[q,1] + dsig2s[q,2]
            sig2s[q,1] = L*N + dsig2s[q,0] + dsig2s[q,1]
            sig2s[q,2] = L*N + dsig2s[q,0]
            m = N/sqrt( sig2s[q,0] )
            
            errs_theor[q,Lidx] = 1.0 - integrate.quad( lambda x: ( ( 0.5*scisp.erfc(-x*sqrt(sig2s[q,0]/(2.0*sig2s[q,1]))) )**(L-1) )*( ( 0.5*scisp.erfc(-x*sqrt(sig2s[q,0]/(2.0*sig2s[q,2]))) )**(D-L) )*exp(-(x-m)*(x-m)/2.0)/sqrt(2.0*pi), -np.inf, np.inf )[0]


    #plt.plot(Ls, errs_theor, '-', color='k')
    #plt.subplot(1,3,Nidx+1)
    for widx in [0,1,2]:#range(3):
        if widx < 2:
            plt.plot(L_theos, errs_theor[widx], '-', color=clrs[widx], alpha=alphas[Nidx])#, ms=5)
    for widx in [0,1,2]:#range(3):
        plt.plot(Ls, aerrs[widx], 'o', color=clrs[widx], ms=7.5, alpha=alphas[Nidx])#, ms=5)
    #plt.ylim(0.0, 0.2)
    #plt.xlim(0, 1500)
    #if Lidx == 0:
    #    plt.ylim(0.0, 0.14)
    #else:
    #    plt.ylim(0.0, 1.1*np.max(errs_theor))
    #plt.xticks([0, 500, 1000], fontsize=16)
    #plt.semilogx()
plt.xlim(0, 15)
plt.ylim(0.0, 1.0)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

#plt.semilogx()
#plt.ylim(0.0, 1.1*errs[2,Llen-1])
#plt.ylim(0.0, 0.55)

plt.show()
svfg.savefig('fig_qbind_dat20g_readout1_D' + str(D) + '_d' + str(d) + '_N' + str(Ns[0]) + '-' + str(Ns[-1]) + '_Tm' + str(Tmax) + '.pdf')



