import h5py
import numpy as np
import pickle
import glob
objects = ['car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train']
for oo in objects:
    filename = '/media/zzs/4TB/qingliu/qing_intermediate/all_K223_res_info/res_info_{0}_train.mat'.format(oo)
    f = h5py.File(filename)
    dic1 = f['res_info']
    len1 = dic1.shape[0]
    
    layer_feature_dist = [None for nn in range(len1)]
    sub_type = [None for nn in range(len1)]
    view_point = [None for nn in range(len1)]
    
    for nn in range(len1):
        dic2 = f[dic1[nn,0]]
        dic21 = dic2["layer_feature_dist"]
        dic21 = np.array(dic21)
        layer_feature_dist[nn] = dic21
        
        dic22 = dic2["sub_type"]
        tmp = ''
        for cc in range(dic22.shape[0]):
            tmp += chr(dic22[cc,0])
        
        sub_type[nn] = tmp
        
        dic23 = dic2["viewpoint"]
        view_point[nn] = dic23["azimuth_coarse"][0,0]
        
    savename = '/media/zzs/4TB/qingliu/qing_intermediate/feat_pickle/res_info_{0}_train.pickle'.format(oo)
    with open(savename, 'wb') as fh:
        pickle.dump([layer_feature_dist, sub_type, view_point], fh)
