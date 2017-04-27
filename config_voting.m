%% configuration file of visual concept project

%% main parameter settings
global category layer_name GPU_id cluster_num


dataset_suffix = 'mergelist_rand';
%% set dirs

load('./ilsvrc_2012_mean.mat');
addpath('/media/zzs/SSD1TB/zzs/modified/caffe/matlab');
model = '/media/zzs/SSD1TB/zzs/surgeried/VGG_ILSVRC_16_layers_deploy_pool5.prototxt';
weights = '/media/zzs/SSD1TB/zzs/surgeried/surgery_weight';
mean_pixel = mean(mean(mean_data, 1), 2);
Caffe.cpu_id = 1;
caffe.reset_all();

% dataset dir
Dataset.img_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Images/%s_imagenet/';
Dataset.anno_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Annotations/%s_imagenet/';
Dataset.sp_anno_dir = '/media/zzs/SSD1TB/zzs/dataset/semantic_file_transfer_support_multiple/%s_imagenet/transfered/';


Data.gt_dir = './intermediate/ground_truth_data/';
Dataset.train_list = fullfile(Data.gt_dir, ['%s_' sprintf('%s_train.txt', dataset_suffix)]);
Dataset.test_list =  fullfile(Data.gt_dir, ['%s_' sprintf('%s_test.txt', dataset_suffix)]);


% where to save dictionary
Dictionary.feature_cache_dir = '/media/zzs/4TB/qing_intermediate/dictionary_imagenet_%s_vgg16_%s_nowarp.mat';
Dictionary.original_dir = '/media/zzs/4TB/qing_intermediate/dictionary_imagenet_%s_vgg16_%s_K%d_norm_nowarp.mat';
Dictionary.new_dir = '/media/zzs/4TB/qing_intermediate/dictionary_imagenet_%s_vgg16_%s_K%d_norm_nowarp_prune_512.mat';

% where to save heatmap features
Heatmap.cache_VC_SP = './intermediate/Heatmap/%s_imagenet_heatmap.mat'; %this for (VC,SP) heatmap
Heatmap.cache_VC_heatmap_feature = './intermediate/Heatmap/%s_imagenet_VC_feature_heatmap.mat'; %this for (VC,SP) pos_set, neg_set
Heatmap.cache_VC_heatmap_likelihood = './intermediate/Heatmap/loglikelihood_%s_imagenet_heatmap.mat'; %this for (VC,SP) log-likelihood

% save scale_path info
cache_patch_info = './intermediate/%s_test_geometry.mat';

% fg mask
Dataset.mask_viewpoint_data = fullfile(Data.gt_dir, ['mask_viewpoint_', category, '_', dataset_suffix, '_%s.mat']);


% these features are got from testing dataset
Feat.cache_dir = '/media/zzs/4TB/qing_intermediate/feat/'; % fullfile(Data.gt_data_dir, 'pos_center');

VC.cache_dir = Feat.cache_dir;
VC.dict_dir = './intermediate/';

Data.train_dir = './intermediate/train/';
Data.test_dir = './intermediate/more_scale_test/';


% where to save VC_SP selection
SP.heatmap_dir = Data.gt_dir;  %

SP.sorted_vc_file = './intermediate/Rank_%s.mat';
SP.selected_vc_file = './intermediate/%s_VC%s.mat';

Voting.model_dir = fullfile(Data.train_dir, sprintf('model_%s_%s', category, dataset_suffix));

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

Apad = Apad_map(layer_name);
Arf = Arf_map(layer_name);
Astride = Astride_map(layer_name);
featDim = featDim_map(layer_name);
offset = offset_map(layer_name); % This offset can fully get rid of out-boundry

% override offset to allow for some close-to-boundary patches
switch layer_name
    case 'pool4'
        offset = 2; % Offset 2 is manually selected for pool4
    case 'pool5'
        offset = 1; % Offset 2 is manually selected for pool5
end

%% deep feature
Feat.layer = layer_name;
Feat.dim = featDim_map(layer_name);

%%
Thresh.cand_values = 0: 0.1: 1.5;
Thresh.value = 0.65;
