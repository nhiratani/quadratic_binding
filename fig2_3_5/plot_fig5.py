#
# Readout of unbinding performance
# under the numerical optimization from distorted sparse Kcomp matrices
#

from math import *
#import sys
import numpy as np
from numpy import linalg as nlg
from scipy import stats as scist

import matplotlib.pyplot as plt
from pylab import cm

climit = 5
clrs = []
for q in range(climit):
    clrs.append( cm.cool( (0.5+q)/float(climit) ) )

N = 48
Tpmax = 100
K = 8
sigrs = [0.3, 0.5, 1.0, 2.0, 3.0]#[0.3, 0.55, 1.0, 1.73, 3.0]
ikmax = 10

slen = len(sigrs)
m_errs = np.zeros((slen,2,Tpmax))
errs = np.zeros((slen,2,ikmax,Tpmax))
for sidx in range(slen):
    sigr = sigrs[sidx]
    for ik in range(ikmax):
        festr = 'data/qbind_Kcomp_basin_errs_N' + str(N) + '_Tpm' + str(Tpmax) + '_K' + str(K) + '_sigr' + str(sigr) + '_ik' + str(ik) + '.txt'
        lidx = 0
        for line in open(festr, 'r'):
            ltmps = line[:-1].split(" ")
            for q in range(2):
                m_errs[sidx,q,lidx] += float(ltmps[1+q])/float(N*ikmax)
                errs[sidx,q,ik,lidx] += float(ltmps[1+q])/float(N)
            lidx += 1

svfg = plt.figure()
plt.axhline(1.0/3.0, ls='--', color='C4', lw=2.0)
plt.axhline(0.2, ls='--', color='C1', lw=2.0)

for sidx in range(slen):
    for ik in range(ikmax):
        plt.plot(range(1,Tpmax+1), errs[sidx,0,ik], color=clrs[sidx], lw=2.0)

plt.xlim(0,26)
plt.ylim(0.0, 0.55)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

svfg.savefig('fig_qbind__Kcomp_basin_errs_readout5_N' + str(N) + '_Tp' + str(Tpmax) + '_K' + str(K) + '_ikm' + str(ikmax) + '.pdf')



