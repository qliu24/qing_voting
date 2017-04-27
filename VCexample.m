load('/media/zzs/4TB/qing_intermediate/dictionary_imagenet_all_vgg16_pool3_K126_norm_nowarp_prune_512.mat')
cd /media/zzs/4TB/qing_intermediate/patch_K126_pool3/
K = 126
num = 100;
for k = 1:K
    dirnm = sprintf('example_K%d',k);
    mkdir(dirnm);
    for i = 1:num
        imwrite(reshape(example{k}(:,i), 44,44,3), sprintf('example_K%d/%d.jpeg', k, i));
    end
end