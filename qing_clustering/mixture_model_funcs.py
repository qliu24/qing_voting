import numpy as np
import sys
from scipy.misc import logsumexp

def compt_weights(lbs, data):
    # lbs: K by N, probability that n belongs to k
    # data: p by N, visible units
    rst_w = np.zeros((lbs.shape[0], data.shape[0])) # K by p
    for kk in range(lbs.shape[0]):
        lb = lbs[kk]
        prob = np.sum(data * lb, axis=1)/np.sum(lb) + 1e-3
        try:
            assert(np.all(prob>0))
            assert(np.all(prob<1))
        except:
            print(np.sum(data * lb, axis=1)[0:10])
            print(np.sum(lb))
            print(prob[0:10])
            sys.exit('error in compt_weights')
            
        rst_w[kk] = np.log(prob/(1-prob))
        
    return rst_w


def compt_lbs(weights, data, lbs_prior=None):
    # weights: K by p
    # data: p by N
    # lbs_prior: K
    K, p = weights.shape
    p2,N = data.shape
    assert(p==p2)
    rst_lbs_log = np.ones((K, N))*-1
    for kk in range(K):
        ww = weights[kk]
        term1 = np.sum(data.T * ww, axis=1)
        term2 = -np.sum(logsumexp(np.stack([ww.reshape(1,-1), np.zeros((1, p))]), axis=0))
        if lbs_prior is not None:
            term3 = np.log(lbs_prior[kk])
        else:
            term3 = 0.
            
        # print('terms')
        # print(term1[0:10], term2, term3)
        rst_lbs_log[kk] = term1+term2+term3
        
    # print('rst_lbs_log')
    # print(rst_lbs_log[:,0:10])
    rst_lbs = np.exp(rst_lbs_log - logsumexp(rst_lbs_log, axis=0))
    rst_log_marg = np.sum(logsumexp(rst_lbs_log, axis=0))/N
    
    return (rst_lbs, rst_log_marg)

