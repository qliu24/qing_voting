from joblib import Parallel, delayed
import pickle
import numpy as np
from vcdist_funcs import *
import time

paral_num=8
file_path = '/media/zzs/4TB/qing_intermediate/feat_pickle/'
savename = file_path + 'car_aeroplane_train_simmat.pickle'
magic_thh = 0.7

fname = file_path + 'res_info_car_train.pickle'
with open(fname, 'rb') as fh:
    layer_feature_dist1, sub_type1, view_point1 = pickle.load(fh)
    
fname = file_path + 'res_info_aeroplane_train.pickle'
with open(fname, 'rb') as fh:
    layer_feature_dist2, sub_type2, view_point2 = pickle.load(fh)
    
fname = file_path + 'res_info_train_train.pickle'
with open(fname, 'rb') as fh:
    layer_feature_dist3, sub_type3, view_point3 = pickle.load(fh)
    

print(len(sub_type1))
print(len(sub_type2))
print(len(sub_type3))

N1 = len(sub_type1)
N2 = len(sub_type2)
N3 = len(sub_type3)
# N1 = 100
# N2 = 100

N = N1+N2+N3
types = np.append(['car' for nn in range(N1)], ['aeroplane' for nn in range(N2)])
types = np.append(types, ['train' for nn in range(N3)])

idx1 = np.argsort(view_point1[0:N1])
idx2 = np.argsort(view_point2[0:N2])
idx3 = np.argsort(view_point3[0:N3])

view_point = np.append(np.array(view_point1)[idx1], np.array(view_point2)[idx2])
view_point = np.append(view_point, np.array(view_point3)[idx3])

layer_feature_b = [None for nn in range(N)]
for nn in range(N1):
    layer_feature_b[nn] = (layer_feature_dist1[idx1[nn]]<magic_thh).astype(int)
    
for nn in range(N1, N1+N2):
    layer_feature_b[nn] = (layer_feature_dist2[idx2[nn-N1]]<magic_thh).astype(int)
    
for nn in range(N1+N2, N):
    layer_feature_b[nn] = (layer_feature_dist3[idx3[nn-N1-N2]]<magic_thh).astype(int)

print('Start compute sim matrix...', flush=True)
_s = time.time()

mat_dis1 = np.ones((N,N))
mat_dis2 = np.ones((N,N))
for nn1 in range(N-1):
    print(nn1, end=' ', flush=True)
    inputs = [(layer_feature_b[nn1], layer_feature_b[nn2]) for nn2 in range(nn1+1,N)]
    para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_sym2)(i) for i in inputs))
    mat_dis1[nn1, nn1+1:N] = para_rst[:,0]
    mat_dis2[nn1, nn1+1:N] = para_rst[:,1]

_e = time.time()
print((_e-_s)/60)
print(mat_dis1.shape)
print(mat_dis1[0])
print(mat_dis2.shape)
print(mat_dis2[0])

with open(savename, 'wb') as fh:
    pickle.dump([mat_dis1, mat_dis2], fh)

