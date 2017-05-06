Data.root_dir2 = '/media/zzs/4TB/qingliu/qing_intermediate/';
dir_det_result = fullfile(Data.root_dir2, 'result');
model_type = 'mix';
dataset_suffix = 'mergelist_rand';
set_type = 'test';

if strcmp(model_type, 'single')
    file_det_result_all = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_single.mat', 'all', dataset_suffix, set_type));
elseif strcmp(model_type, 'mix')
    file_det_result_all = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_mix.mat', 'all', dataset_suffix, set_type));
else
    error('Error: unknown model_type');
end

objects = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};

det_all = cell(0,1)
for i = 1:numel(objects)
    category = objects{i};
    
    if strcmp(model_type, 'single')
        file_det_result = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_single.mat', category, dataset_suffix, set_type));
    elseif strcmp(model_type, 'mix')
        file_det_result = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_mix.mat', category, dataset_suffix, set_type));
    else
        error('Error: unknown model_type');
    end
    
    assert( exist(file_det_result, 'file') > 0 );
    load(file_det_result, 'det_all');
    for n = 1:length(det)
        det{n}.cat = category;
    end
    det_all = [det_all;det];
end

save(file_det_result_all, 'det_all', '-v7.3');