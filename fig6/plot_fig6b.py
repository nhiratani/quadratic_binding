#
# Readout of unbinding performance
#

from math import *
import numpy as np
from numpy import linalg as nlg

import matplotlib.pyplot as plt
from pylab import cm

N = 48
L = 3
Tpmax = 300
ik = 0

fpstr = 'data/qbind_opt_PQR_L_tensor_N' + str(N) + '_L' + str(L) + '_Tpm' + str(Tpmax) + '_ik' + str(ik) + '.txt'
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
A = np.dot( nlg.inv(P[mu,:,:]), np.dot(u0, np.diag(np.sqrt(s0))) )

svfg = plt.figure()
qmax = 5
Pqs = np.zeros((qmax,N,N))
for q in range(qmax):
    Pqs[q,:,:] = np.dot(np.transpose(u0), np.dot(P[q,:,:], A))
Pqmax = np.amax(Pqs)

Pmax = np.amax(P)
for q in range(qmax):
    plt.subplot(1,qmax,q+1)
    im = plt.imshow(Pqs[q], vmin=-Pqmax, vmax=Pqmax, cmap='bwr')
    plt.colorbar(im, fraction=0.046, pad=0.04)
plt.show()
svfg.savefig('fig_qbind_readout_Pbar_matrix_N' + str(N) + '_L' + str(L) + '_Tp' + str(Tpmax) + '_ik' + str(ik) + '.pdf')



