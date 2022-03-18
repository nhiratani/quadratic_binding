#
# Readout of the structures in P, Q, R after a numerical optimization
#

from math import *
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg
from scipy import stats as scist

import matplotlib.pyplot as plt
from pylab import cm

N = 48
ikmax = 10
Tpmax = 100

Q = nrnd.normal(0.0, 1.0/sqrt(float(N*N)), (N,N,N))
R = nrnd.normal(0.0, 1.0/sqrt(float(N*N)), (N,N,N))
dQR_init = np.sum( np.multiply(Q-R, Q-R) )
print( dQR_init )

m_errs = np.zeros((2,Tpmax+1))
errs = np.zeros((2,ikmax,Tpmax+1))

m_norms = np.zeros((2,Tpmax+1))
norms = np.zeros((2,ikmax,Tpmax+1))

m_syms = np.zeros((2,Tpmax+1))
syms = np.zeros((2,ikmax,Tpmax+1))
for ik in range(ikmax):
    festr = 'data/qbind_opt_PQR_err_N' + str(N) + '_Tpm' + str(Tpmax) + '_ik' + str(ik) + '.txt'
    lidx = 0
    for line in open(festr, 'r'):
        ltmps = line[:-1].split(" ")
        for q in range(2):
            m_errs[q,lidx] += float(ltmps[1+q])/float(N*ikmax)
            errs[q,ik,lidx] += float(ltmps[1+q])/float(N)
        for q in range(2):
            m_norms[q,lidx] += float(ltmps[3+q])/float(N*ikmax)
            norms[q,ik,lidx] += float(ltmps[3+q])/float(N)
        for q in range(2):
            m_syms[q,lidx] += float(ltmps[5+q])/float(N*ikmax)
            syms[q,ik,lidx] += float(ltmps[5+q])/float(N)
        lidx += 1

svfg = plt.figure()
for ik in range(ikmax):
    plt.plot(range(0,Tpmax+1), norms[0,ik], color='C0', lw=1.0)
    plt.plot(range(0,Tpmax+1), norms[1,ik], color='C1', lw=1.0)
    plt.plot(range(0,Tpmax+1), syms[0,ik], color='C2', lw=1.0)

plt.xlim(0,30)
plt.semilogy()
plt.ylim(0.0000000001, 10.0)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

svfg.savefig('fig_qbind_opt_PQR_dQR_dQP_readout2c_N' + str(N) + '_Tp' + str(Tpmax) + '_ikm' + str(ikmax) + '.pdf')


