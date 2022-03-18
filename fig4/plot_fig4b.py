#
# Readout of unbinding performance
#

from math import *
#import sys
import numpy as np
from scipy import stats as scist

import matplotlib.pyplot as plt
#from pylab import cm

clrs = ['C7', 'C4', 'C1']

N = 128
Ks = [1,2,4,8,16,32,64]
Tmax = 10000

Klen = len(Ks)
errs = np.zeros((Klen))
for Kidx in range(Klen):
    K = Ks[Kidx]
    festr = 'data/qbind_sKcomp_HRR_error_N' + str(N) + '_K' + str(K) + '_Tm' + str(Tmax) + '.txt'
    for line in open(festr, 'r'):
        ltmp = line[:-1].split(" ")
        errs[Kidx] = float(ltmp[0])

svfg = plt.figure()
K_theors = np.arange(1, 8, 0.01)
err_theors = []
for K_theor in K_theors:
    err_theors.append( 2.0/(2.0 + K_theor) )
plt.plot(K_theors, err_theors, '-', color='k')
plt.plot(Ks[:4], errs[:4], 'o', color='k', ms=10)
plt.plot(Ks[3:], errs[3:], 'o--', color='k', ms=10)

plt.semilogx()
plt.ylim(0.0, 0.9)
plt.xlim(0.8, 80)
plt.xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64], fontsize=16)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=16)
plt.show()
svfg.savefig('fig_qbind_sKcomp_HRR_error_readout2_K_2-64' + '_N' + str(N) + '_Tm' + str(Tmax) + '.pdf')



