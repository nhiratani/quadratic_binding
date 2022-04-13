#
# Readout of unbinding performance
#

from math import *
import numpy as np
import matplotlib.pyplot as plt

N = 48
L = 3
ikmax = 10
Tpmax = 300

m_errs = np.zeros((2,Tpmax+1))
errs = np.zeros((2,ikmax,Tpmax+1))

m_norms = np.zeros((2,Tpmax+1))
norms = np.zeros((2,ikmax,Tpmax+1))
for ik in range(ikmax):
    festr = 'data/qbind_opt_PQR_err_N' + str(N) + '_L' + str(L) + '_Tpm' + str(Tpmax) + '_ik' + str(ik) + '.txt'
    lidx = 0
    for line in open(festr, 'r'):
        ltmps = line[:-1].split(" ")
        for q in range(2):
            m_errs[q,lidx] += float(ltmps[1+q])/float(N*ikmax)
            errs[q,ik,lidx] += float(ltmps[1+q])/float(N)
        for q in range(2):
            m_norms[q,lidx] += float(ltmps[3+q])/float(N*ikmax)
            norms[q,ik,lidx] += float(ltmps[3+q])/float(N)
        lidx += 1

svfg = plt.figure()
plt.axhline( ((L-1)*8.0+2.0)/(L*8.0+2.0), c='k', ls='--')
plt.axhline( (L/float(L+1)), c='gray', ls='--')
for ik in range(ikmax):
    plt.plot(range(0,Tpmax+1), errs[0,ik], color='k', lw=1.0)

plt.xlim(0,100)
plt.ylim(0.5, 1.0)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

svfg.savefig('fig_qbind_error_readout2_la_N' + str(N) + '_L' + str(L) + '_Tp' + str(Tpmax) + '_ikm' + str(ikmax) + '.pdf')
