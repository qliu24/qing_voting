import h5py
import numpy as np
import pickle
import glob
from scipy.stats import spearmanr
import scipy.io as sio

objects=['car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train']
magic_thh = 0.7

for oo in objects:
    print('working on {0}'.format(oo))
    filename = '/media/zzs/4TB/qingliu/qing_intermediate/all_K216_res_info/res_info_{0}_train.mat'.format(oo)
    f = h5py.File(filename)
    dic1 = f['res_info']
    len1 = dic1.shape[0]
    
    layer_feature_dist = [None for nn in range(len1)]
    for nn in range(len1):
        dic2 = f[dic1[nn,0]]
        dic21 = dic2["layer_feature_dist"]
        dic21 = np.array(dic21)
        layer_feature_dist[nn] = dic21
    
    N=len1
    layer_feature_b = [None for nn in range(N)]
    for nn in range(N):
        layer_feature_b[nn] = (layer_feature_dist[nn]<magic_thh).astype(int)
        
    # VC num
    max_0 = max([layer_feature_b[nn].shape[0] for nn in range(N)])
    # width
    max_1 = max([layer_feature_b[nn].shape[1] for nn in range(N)])
    # height
    max_2 = max([layer_feature_b[nn].shape[2] for nn in range(N)])
    
    all_bg_b = np.zeros((max_0, max_1, max_2))
    for nn in range(N):
        vnum, ww, hh = layer_feature_b[nn].shape
        assert(vnum == max_0)
        
        diff_w1 = int((max_1-ww)/2)
        diff_w2 = int(max_1-ww-diff_w1)
        assert(max_1 == diff_w1+diff_w2+ww)
        
        diff_h1 = int((max_2-hh)/2)
        diff_h2 = int(max_2-hh-diff_h1)
        assert(max_2 == diff_h1+diff_h2+hh)
        
        padded = np.pad(layer_feature_b[nn], ((0,0),(diff_w1, diff_w2),(diff_h1, diff_h2)), 'constant', constant_values=0)
        all_bg_b += padded
        if nn%100==0:
            print(nn)
        
    probs = all_bg_b/N + 1e-3
    weights = np.log(probs/(1.-probs))
    sio.savemat('/media/zzs/4TB/qingliu/qing_intermediate/unary_weights/{0}_train.mat'.format(oo), mdict={'weight': weights})