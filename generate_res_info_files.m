function generate_res_info_files(set_type, config_file)
try
    eval(config_file);
catch
    keyboard;
end

dir_img = sprintf(Dataset.img_dir, category);
dir_anno = sprintf(Dataset.anno_dir, category);

switch set_type
    case 'train'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_train.mat', category, dataset_suffix));
        file_list = sprintf(Dataset.train_list, category);
    case 'test'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_test.mat', category, dataset_suffix));
        file_list = sprintf(Dataset.test_list, category);
    otherwise
        error('Error: unknown set_type;\n');
end  

assert(exist(file_cache_VC_data, 'file') > 0);
load(file_cache_VC_data);
assert(exist('r_set', 'var')>0);
assert(exist('feat_set', 'var')>0);

assert(exist(file_list, 'file') > 0);
file_ids = fopen(file_list, 'r');
img_list = textscan(file_ids, '%s %d');
img_num = length(img_list{1});

res_info = cell(1,img_num);
for n=1:img_num
    file_img = sprintf('%s/%s.JPEG', dir_img, img_list{1}{n});
    file_anno = sprintf('%s/%s.mat', dir_anno, img_list{1}{n});
    anno = load(file_anno);
    anno = anno.record;
    
    res_info{n}.img = imread(file_img);
    res_info{n}.layer_feature_dist = r_set{n};
    res_info{n}.layer_feature_ori = feat_set{n};
    res_info{n}.sub_type = anno.objects(img_list{2}(n)).subtype;
    res_info{n}.viewpoint = anno.objects(img_list{2}(n)).viewpoint;
    if mod(n,100) == 0
        disp(n);
    end 
end

VC.res_info = '/media/zzs/4TB/qing_intermediate/all_K216_res_info/res_info_%s_%s.mat';
save(sprintf(VC.res_info, category, set_type), 'res_info', '-v7.3');

end % end of function

qin@semantic-All-Series:~/voting/qing_voting$ cat matrixDist.m 
function result = matrixDist(data, dictionary)
% data: K * N
% dictionary: K * M
% result M * N
%% method 1
temp = bsxfun(@plus, sum(data.^2, 1), sum(dictionary.^2, 1)');
result = temp - 2*dictionary'*data;

% %% method 2 for comparison
% [K, N] = size(data);
% [K, M] = size(dictionary);
% result = zeros(M, N);
% for n = 1:N
%     error = sum(bsxfun(@minus, data(:, n), dictionary).^2, 1);
%     result(:, n) = error';
% end

end