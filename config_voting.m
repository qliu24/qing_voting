%% configuration file of visual concept project

%% main parameter settings
global category layer_name GPU_id


dataset_suffix = 'mergelist_rand';
%% set dirs

load('./ilsvrc_2012_mean.mat');
addpath('/media/zzs/SSD1TB/zzs/modified/caffe/matlab');
model = '/media/zzs/SSD1TB/zzs/surgeried/VGG_ILSVRC_16_layers_deploy_pool5.prototxt';
weights = '/media/zzs/SSD1TB/zzs/surgeried/surgery_weight';
mean_pixel = mean(mean(image_mean, 1), 2);
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
Dictionary.feature_cache_dir = './intermediate/dictionary/dictionary_imagenet_%s_vgg16_%s_nowarp.mat';
cluster_num = 512;
Dictionary.original_dir = './intermediate/dictionary/dictionary_imagenet_%s_vgg16_%s_K%d_norm_nowarp.mat';
Dictionary.new_dir = './intermediate/dictionary/dictionary_imagenet_%s_vgg16_%s_K%d_norm_nowarp_prune_512.mat';

% where to save heatmap features
Heatmap.cache_VC_SP = './intermediate/Heatmap/%s_imagenet_heatmap.mat'; %this for (VC,SP) heatmap
Heatmap.cache_VC_heatmap_feature = './intermediate/Heatmap/%s_imagenet_VC_feature_heatmap.mat'; %this for (VC,SP) pos_set, neg_set
Heatmap.cache_VC_heatmap_likelihood = './intermediate/Heatmap/loglikelihood_%s_imagenet_heatmap.mat'; %this for (VC,SP) log-likelihood

% save scale_path info
cache_patch_info = './intermediate/%s_test_geometry.mat';



% these features are got from testing dataset
Feat.cache_dir = './intermediate/feat/'; % fullfile(Data.gt_data_dir, 'pos_center');

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

%% VC
VC.layer = layer_name;

switch category
    case 'car'
        VC.num = cluster_car;
    case 'aeroplane'
        VC.num = cluster_aeroplane;
    case 'bus'
        VC.num = cluster_bus;
    case 'bicycle'
        VC.num = cluster_bicycle;
    case 'train'
        VC.num = cluster_train;
    case 'motorbike'
        VC.num = cluster_motorbike;
end


%% SP
% car: 39, aeroplane: 20, bicycle: 14, motorbike: 13, train: 16, bus: 10
switch category
    case 'car'
        SP.num = 39;
    case 'aeroplane'
        SP.num = 20;
    case 'bicycle'
        SP.num = 14;
    case 'motorbike'
        SP.num = 13;
    case 'train'
        SP.num = 16;
    case 'bus'
        SP.num = 10;
    otherwise
        error('Unknown category!\n');
end
SP.patch_size = [100 100];
SP.det_thresh = 0;
SP.feat_interp_type = 'nearest_neighbour';                                 % the interpolation type of deep features for SPs: {'nearest_neighbour', 'bilinear'}

%% SP_Pair
SPPair.by_pass = true;
switch category
    case 'car'
        SPPair.list = [1, 10;
            1, 12;
            1, 17;
            1, 19;
            10, 10;
            12, 12;
            10, 12;
            17, 17;
            17, 19;
            18, 20;
            18, 18;
            23, 18;
            23, 20;
            24, 17;
            24, 19;
            34, 17;
            34, 19;
            1,  34;
            12, 35;
            12, 14;
            1, 35;
            10, 14;
            10, 35;
            1,  14];
        SPPair.list = sort(SPPair.list, 2);
        SPPair.list = unique(SPPair.list, 'rows');

        SPPair.num = size(SPPair.list, 1);
        SPPair.unique_list = [1, 17];
        SPPair.max_dist_voting = 256;
        %     otherwise
        %         error('Unknown category!\n');
end


%%
SPTriplet.by_pass = true;
switch category
    case 'car'
        SPTriplet.list = [1, 10, 35;
            1, 10, 12;
            1, 17, 34;
            10, 12, 35;
            14, 10, 12];
        SPTriplet.num = size(SPTriplet.list, 1);

        %     otherwise
        %         error('Unknown category!\n');
end


%% Voting
Voting.by_pass = true;
Voting.vis = false;

Voting.model_type = 'heatmap';
Voting.heatmap.mode = 'nonparam_prior';                                    % the mode of processing score map: {'gauss_smooth', 'nonparam_prior'}
Voting.heatmap.resolution = 0.5;
Voting.heatmap.score_thresh = 0;
Voting.heatmap.d_max_pool = d_max_pool;
Voting.heatmap.bin_siz_max_pool = bin_siz_max_pool;

% for mode 'gauss_smooth'
Voting.heatmap.n_gauss_std = 3;
% for mode 'nonparam_prior'
Voting.heatmap.nonparam_beta = 0.7;
Voting.heatmap.nonparam_tradeoff_weight_sp_pair = 1;                       % the tradeoff weight of geometry term for SP pairs
Voting.heatmap.interpolation_method = 'bilinear';

%% NMS
NMS.score_ratio = 0.2;
%NMS.bbox_ratio = 0.15;
NMS.score_ratio_cihang = 0;
NMS.bbox_ratio_cihang = 0;
% NMS.thresh_dir = './evaluation/car_imagenet_SP_nms_thresh.mat';

%% Evaluate
Eval.dist_thresh = 56;
Eval.vis_scoremap_with_det = false;

%%
Voting.scale_base = 1.6;
Voting.select_sp_num = 39; % how many SPs to use in detection and evaluation.

%% sp_pair
sp_pair_num = 24;

% cihang's occlusion dataset name.

switch category
    case 'car'
        ocld_set = {'car_random_0.2-0.4_four.mat', ...
            'car_random_0.2-0.4_three.mat', ...
            'car_random_0.2-0.4_two.mat', ...
            'car_random_0.4-0.6_four.mat', ...
            'car_random_0.4-0.6_three.mat', ...
            'car_random_0.4-0.6_two.mat', ...
            'car_random_0.6-0.8_four.mat', ...
            'car_random_0.6-0.8_three.mat', ...
            'car_random_0.6-0.8_two.mat'};
end

%Multi-scale
%MS.scale=[224, 320, 480, 640, 864];
MS.scale=[224, 272, 320, 400, 480, 560, 640, 752, 864];

