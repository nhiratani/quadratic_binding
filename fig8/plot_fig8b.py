#
# Readout of unbinding performance with a dictionary
#

from math import *
#import sys
import numpy as np

from scipy import stats as scist
from scipy import special as scisp
from scipy import integrate as integrate

import matplotlib.pyplot as plt

clrs = ['C1', 'C0', 'C3'] #(ext-Oct, Tensor-HRR, Random-quad)

Ns = [96, 192, 384]
d = 1
Ls = [5, 10, 20]
Ds = range(250, 5001, 250)
Tmax = 10000

Llen = len(Ls)
Dlen = len(Ds)

Dtheos = range(25, 5001, 25)

svfg = plt.figure()
for Lidx in range(Llen):
    N = Ns[Lidx]
    L = Ls[Lidx]
    aerrs = np.zeros((3, Dlen))
    for Didx in range(Dlen):
        D = Ds[Didx]
        festr = 'data/qbind_dict_ext_sKcomp_tensor_HRR_random_error_N' + str(N) + '_d' + str(d) + '_L' + str(L) + '_D' + str(D) + '_Tm' + str(Tmax) + '.txt'
        
        lidx = 0
        for line in open(festr, 'r'):
            ltmp = line[:-1].split(" ")
            aerrs[lidx, Didx] = float(ltmp[1])
            lidx += 1

    K = 8
    dsig2s = np.array([[2.0/K, 2.0/K, 1.0],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0]])

    sig2s = np.zeros((3,3))
    for q in range(3):
        sig2s[q,0] = (L + dsig2s[q,0] + dsig2s[q,1] + dsig2s[q,2])/float(N)
        sig2s[q,1] = (L + dsig2s[q,0] + dsig2s[q,1])/float(N)
        sig2s[q,2] = (L + dsig2s[q,0])/float(N)

    errs_theor = np.zeros((3, len(Dtheos)))
    for q in range(3):
        for Didx in range( len(Dtheos) ):
            D = Dtheos[Didx]
            m = 1.0/sqrt( sig2s[q,0] )

            errs_theor[q,Didx] = 1.0 - integrate.quad( lambda x: ( ( 0.5*scisp.erfc(-x*sqrt(sig2s[q,0]/(2.0*sig2s[q,1]))) )**(L-1) )*( ( 0.5*scisp.erfc(-x*sqrt(sig2s[q,0]/(2.0*sig2s[q,2]))) )**(D-L) )*exp(-(x-m)*(x-m)/2)/sqrt(2.0*pi), -np.inf, np.inf )[0]

    plt.subplot(1,3,Lidx+1)
    for widx in range(3):
        if widx < 2:
            plt.plot(Dtheos, errs_theor[widx], '-', color=clrs[widx])
        plt.plot(Ds, aerrs[widx], 'o', color=clrs[widx], ms=5)
    plt.ylim(0.0, 0.4)
    plt.xlim(0, 5100)
    if Lidx == 0:
        plt.yticks([0.0,0.1,0.2,0.3,0.4])
    else:
        plt.yticks([])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

plt.show()
svfg.savefig('fig_qbind_dict_ext_readout7b_D' + str(Ds[0]) + '_' + str(Ds[-1]) + '_d' + str(d) + '_N' + str(Ns[0]) + '-' + str(Ns[-1]) + '_Tm' + str(Tmax) + '.pdf')



