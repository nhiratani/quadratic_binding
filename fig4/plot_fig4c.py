#
# Readout of unbinding performance
#

from math import *
#import sys
import numpy as np
#from numpy import random as nrnd
#from numpy import linalg as nlg
from scipy import stats as scist

import matplotlib.pyplot as plt
#from pylab import cm

clrs = ['C7', 'C6', 'C4', 'C1']

# K = 0 corresponds to Holographic
Ns = [8,16,32,64,128,256]
Ks = [0,1,2,4,8]#[0, 2, 4, 8]
Tmax = 10000

Klen = len(Ks); Nlen = len(Ns)
errs = np.zeros((Klen, Nlen))
for Kidx in range(Klen):
    K = Ks[Kidx]
    for Nidx in range(Nlen):
        N = Ns[Nidx]
        festr = 'data/qbind_norm_sKcomp_HRR_error_N' + str(N) + '_K' + str(K) + '_Tm' + str(Tmax) + '.txt'
        for line in open(festr, 'r'):
            ltmp = line[:-1].split(" ")
            errs[Kidx, Nidx] = float(ltmp[0])

svfg = plt.figure()

for Kidx in range(Klen):
    K = Ks[Kidx]
    if K == 0:
        xs = range(2,350)
        ys = 1.0 + np.divide( 2.0*np.ones((len(xs))), xs )
        plt.plot(xs, ys,  ls='-', color='C0')
        plt.plot(Ns, errs[Kidx], 'o', color='C0', ms=10)
    else:
        plt.axhline(2.0/K, color=clrs[Kidx-1])
        plt.plot(Ns, errs[Kidx], 'o', color=clrs[Kidx-1], ms=10)

plt.semilogx()
plt.xlim(6,310)
plt.ylim(0.0, 2.1)
plt.xticks([10, 30, 100, 300], [10, 30, 100, 300], fontsize=16)
plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0], fontsize=16)
plt.show()
svfg.savefig('fig_qbind_norm_sKcomp_HRR_error_readout1_K_2-4-8' + '_Tm' + str(Tmax) + '.pdf')



