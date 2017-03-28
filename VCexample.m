load('/media/zzs/4TB/qing_intermediate/dictionary_imagenet_all2_vgg16_pool4_K216_norm_nowarp_prune_512.mat')
cd /media/zzs/4TB/qing_intermediate/patch_K216/
K = 216
num = 100
for k = 1:K
    dirnm = sprintf('example_K%d',k);
    mkdir(dirnm);
    for i = 1:num
        imwrite(reshape(example{k}(:,i), 100,100,3), sprintf('example_K%d/%d.jpeg', k, i));
    end
end