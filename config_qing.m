global category model_category;
dataset_suffix = 'mergelist_rand';
layer_name = 'pool4';
% category = 'bus';
% model_category = 'aeroplane';
set_type = 'test';

model_type = 'mix'; % or single
model_suffix = sprintf('%s.mat', model_type);

%% feature parameter
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
offset = offset_map(layer_name); % This offset can fully get rid of out-boundry
% override offset to allow for some close-to-boundary patches
offset = 2; % Offset 2 is manually selected for pool4
feat_dim = featDim_map(layer_name);

VC.dict_dir = './intermediate/dictionary/';
VC.layer = layer_name;
VC.num = 216;

file_VC_dict = fullfile(VC.dict_dir, sprintf('dictionary_imagenet_%s_vgg16_%s_K%i_norm_nowarp_prune_%i.mat', 'all', VC.layer, VC.num, feat_dim));

%% Caffe parameter
% Caffe.dir = '/media/zzs/5TB/tmp/caffe/';
% addpath(fullfile(Caffe.dir, 'matlab'));
% 
% load('./ilsvrc_2012_mean.mat');
% model = '/media/zzs/SSD1TB/zzs/surgeried/VGG_ILSVRC_16_layers_deploy_pool5.prototxt';
% weights = '/media/zzs/SSD1TB/zzs/surgeried/surgery_weight';
% mean_pixel = mean(mean(mean_data, 1), 2);
% Caffe.gpu_id = 1;


%% set image pathes
Dataset.img_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Images/%s_imagenet/';
Dataset.anno_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Annotations/%s_imagenet/';
Data.gt_dir = './intermediate/ground_truth_data/';
Dataset.train_list = fullfile(Data.gt_dir, ['%s_' sprintf('%s_train.txt', dataset_suffix)]);
Dataset.test_list =  fullfile(Data.gt_dir, ['%s_' sprintf('%s_test.txt', dataset_suffix)]);
dir_img = sprintf(Dataset.img_dir, category);

Data.root_dir = './intermediate/data/';
Data.root_dir2 = '/media/zzs/4TB/qingliu/qing_intermediate/';
dir_feat_bbox_proposals = fullfile(Data.root_dir2, 'feat');

Model.dir = '/media/zzs/4TB/qingliu/qing_intermediate/unary_weights/';

% Model_file_bg = fullfile(Model.dir, 'all_train_bg2.mat');

dir_det_result = fullfile(Data.root_dir2, 'result');
MkdirIfMissing(dir_det_result);
dir_perf_eval = fullfile(Data.root_dir2, 'eval_obj_det');
MkdirIfMissing(dir_perf_eval);


if strcmp(model_type, 'single')
    Model_file = fullfile(Model.dir, sprintf('%s_train.mat', model_category));
elseif strcmp(model_type, 'mix')
    Model_file = fullfile(Model.dir, sprintf('%s_K4_softstart.mat', model_category));
else
    error('Error: unknown model_type');
end

if strcmp(model_category, 'bg1')
    Model_file = fullfile(Model.dir, 'car_train_bg.mat');
elseif strcmp(model_category, 'bg2')
    Model_file = fullfile(Model.dir, 'bus_train_bg.mat');
elseif strcmp(model_category, 'bg3')
    Model_file = fullfile(Model.dir, 'aeroplane_train_bg.mat');
elseif strcmp(model_category, 'bg4')
    Model_file = fullfile(Model.dir, 'train_train_bg.mat');
elseif strcmp(model_category, 'bg5')
    Model_file = fullfile(Model.dir, 'bicycle_train_bg.mat');
elseif strcmp(model_category, 'bg6')
    Model_file = fullfile(Model.dir, 'motorbike_train_bg.mat');
end

temp_dim = containers.Map;
temp_dim('car') = [17 55 216];
temp_dim('bus') = [25 52 216];
temp_dim('aeroplane') = [22 56 216];
temp_dim('train') = [35 56 216];
temp_dim('bicycle') = [30 33 216];
temp_dim('motorbike') = [37 35 216];

file_det_result = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                   model_category, category, dataset_suffix, set_type, model_suffix));
file_det_result_all = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       model_category, 'all', dataset_suffix, set_type, model_suffix));
file_det_result_all_bg1 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bg', category, dataset_suffix, set_type, 'single.mat'));
file_det_result_all_bg2 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'bg2', category, dataset_suffix, set_type, 'single.mat'));
file_det_result_all_bg3 = fullfile(dir_det_result, sprintf('props_det_%s_%s_%s_%s_%s', ...
                                                       'car', category, dataset_suffix, set_type, 'mix.mat'));

file_det_result_all_bg = cell(3,1);
file_det_result_all_bg{1} = file_det_result_all_bg1;
file_det_result_all_bg{2} = file_det_result_all_bg2;
file_det_result_all_bg{3} = file_det_result_all_bg3;

file_perf_eval = fullfile(dir_perf_eval, sprintf('eval_%s_%s_%s_%s', ...
                                                  model_category, dataset_suffix, set_type, model_suffix));
file_perf_eval_all = fullfile(dir_perf_eval, sprintf('eval_%s_%s_%s_all_%s', ...
                                                  model_category, dataset_suffix, set_type, model_suffix));

file_gt_obj_anno = fullfile(dir_perf_eval, sprintf('gt_anno_%s_%s_%s', model_category, dataset_suffix, set_type));

