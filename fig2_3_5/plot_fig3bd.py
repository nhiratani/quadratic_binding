#
# Readout ofthe optimized unbinding matrices Q
#

from math import *
import numpy as np
from numpy import linalg as nlg
from scipy import stats as scist

import matplotlib.pyplot as plt
from pylab import cm

N = 48
ik = 0
Tpmax = 100

fpstr = 'data/qbind_opt_PQR_tensor_N' + str(N) + '_Tpm' + str(Tpmax) + '_ik' + str(ik) + '.txt'
P = np.zeros((N,N,N))
Q = np.zeros((N,N,N))

iidx = 0; jidx = 0; kidx = 0
for line in open(fpstr, 'r'):
    ltmps = line[:-1].split(" ")
    P[iidx,jidx,kidx] = float(ltmps[0])
    Q[iidx,jidx,kidx] = float(ltmps[1])
    
    kidx += 1
    if kidx == N:
        kidx = 0; jidx += 1
        if jidx == N:
            jidx = 0; iidx += 1

mu = 0
u0, s0, vh0 = nlg.svd( np.dot(P[mu,:,:], np.transpose(Q[mu,:,:])) )
v0 = np.transpose(vh0)
B = np.dot( nlg.inv(Q[mu,:,:]), np.dot(v0, np.diag(np.sqrt(s0))) )

svfg = plt.figure()
qmax = 5
Qqs = np.zeros((qmax,N,N))
for q in range(qmax):
    Qqs[q,:,:] = np.dot( vh0, np.dot(Q[q,:,:], B) )
Qqmax = np.amax(Qqs)

Qmax = np.amax(Q)
for q in range(qmax):
    plt.subplot(2,qmax,q+1)
    plt.imshow(Q[q], vmin=-Qmax, vmax=Qmax, cmap='bwr')
    plt.colorbar()
    
    plt.subplot(2,qmax,qmax + q+1)
    plt.imshow(Qqs[q], vmin=-Qqmax, vmax=Qqmax, cmap='bwr')
    plt.colorbar()
plt.show()
svfg.savefig('fig_qbind_optPQR_readout3Q_N' + str(N) + '_Tp' + str(Tpmax) + '_ik' + str(ik) + '.pdf')

