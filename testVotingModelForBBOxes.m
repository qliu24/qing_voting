% test voting models for bounding box proposals 
% created by Jun Zhu @JHU, on 11/11/2016.

function testVotingModelForBBOxes(set_type)
%%
fprintf('test voting models for bounding box proposals on "%s" set ...\n', set_type);

%% Caffe parameter
caffe_dim = 224; % caffe input dimension in deploy protobuf
layer_set = {'pool1', 'pool2', 'pool3', 'pool4', 'pool5'};
% in original image input space
Apad_set = [2, 6, 18, 42, 90]; % padding size
Astride_set = [2, 4, 8, 16, 32]; % stride size
featDim_set = [64, 128, 256, 512, 512]; % feature dimension
Arf_set = [6, 16, 44, 100, 212];
offset_set = ceil(Apad_set./Astride_set);

Apad_map = containers.Map(layer_set, Apad_set);
Arf_map = containers.Map(layer_set, Arf_set);
Astride_map = containers.Map(layer_set, Astride_set);
featDim_map = containers.Map(layer_set, featDim_set);
offset_map = containers.Map(layer_set, offset_set);

dataset_suffix = 'mergelist_rand';
layer_name = 'pool4';
category = 'car';
%%
Apad = Apad_map(layer_name);
Arf = Arf_map(layer_name);
Astride = Astride_map(layer_name);

% set image pathes
Dataset.img_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Images/%s_imagenet/';
Data.gt_dir = './intermediate/ground_truth_data/';
Dataset.train_list = fullfile(Data.gt_dir, ['%s_' sprintf('%s_train.txt', dataset_suffix)]);
Dataset.test_list =  fullfile(Data.gt_dir, ['%s_' sprintf('%s_test.txt', dataset_suffix)]);

switch set_type
    case 'train'
        file_list = sprintf(Dataset.train_list, category);
    case 'test'
        file_list = sprintf(Dataset.test_list, category);
    otherwise
        error('Error: unknown set_type');
end   
assert(exist(file_list, 'file') > 0);
file_ids = fopen(file_list, 'r');
img_list = textscan(file_ids, '%s %d');
img_num = length(img_list{1});

%%
% load model
Model.dir = '/media/zzs/4TB/qingliu/qing_intermediate/unary_weights/';
Model_file = fullfile(Model.dir, 'cars_K4_softstart.mat');
load(Model_file);
% weight for unary models
% mixture_weights, mixture_priors for mixture models
% weight_obj = permute(weight,[3,2,1]);
weight_objs = cell(size(mixture_weights,1),1);
log_priors = cell(size(mixture_weights,1),1);
for mm=1:size(mixture_weights,1)
    weight_objs{mm} = reshape(mixture_weights(mm,:), 17,55,216);
    log_priors{mm} = log(mixture_priors(mm));
end

Model_file = fullfile(Model.dir, 'car_train_bg.mat');
load(Model_file);
weight_obj_bg = permute(weight,[3,2,1]);

%% compute voting scores for each image
Data.root_dir2 = '/media/zzs/4TB/qingliu/qing_intermediate/';
dir_feat_bbox_proposals = fullfile(Data.root_dir2, 'feat');

num_batch = length(dir(fullfile(dir_feat_bbox_proposals, ...
                                  sprintf('props_feat_%s_%s_%s_*.mat', category, dataset_suffix, set_type))));
                              
det = cell([img_num, 1]);   % det{n} ~ struct('img_path', 'img_siz', 'box', 'box_siz', 'score')                              
n = 0;             
fprintf('compute voting scores ...');
for i = 1: num_batch
    fprintf(' for batch %d of %d:', i, num_batch);
    
    file_cache_feat_batch = fullfile(dir_feat_bbox_proposals, sprintf('props_feat_%s_%s_%s_%d.mat', ...
                                      category, dataset_suffix, set_type, i));
    assert( exist(file_cache_feat_batch, 'file') > 0 );
    
    clear('feat');
    load(file_cache_feat_batch, 'feat');
            
    for cnt_img = 1: length(feat)
        n = n + 1;
        
        det{n}.img_path = feat{cnt_img}.img_path;
        det{n}.img_siz = feat{cnt_img}.img_siz;
        det{n}.box = feat{cnt_img}.box;
        det{n}.box_siz = feat{cnt_img}.box_siz;
                
        num_box = size(feat{cnt_img}.box, 1);        
        det{n}.score = zeros([num_box, 1]);
        for j = 1: num_box            
            det{n}.score(j, 1) = comptScoresM(feat{cnt_img}.r{j}, weight_objs, log_priors) - comptScores(feat{cnt_img}.r{j}, weight_obj_bg);
        end
        
        if mod(cnt_img, 10) == 0
            fprintf(' %d', n);
        end
    end % n: image index in batch
    
    fprintf('\n');
    
end % i: batch index
assert(n == img_num);

%%
dir_det_result = fullfile(Data.root_dir2, 'result');
MkdirIfMissing(dir_det_result);

file_det_result = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s.mat', category, dataset_suffix, set_type));
save(file_det_result, 'det', '-v7.3');

end % end of function

function score = comptScoresM(input, weight_objs, log_priors)
    score_i = zeros(length(weight_objs),1);
    for mm=1:length(weight_objs)
        logllk = comptScores(input, weight_objs{mm});
        score_i(mm) = logllk+log_priors{mm};
    end
    score = logsumexp(score_i);
end

