% extract layer features for bounding-box proposals
function comptFeatForBBoxes(set_type)

fprintf('extract deep network layer features for bounding box proposals on "%s" set ...\n', set_type);

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
set_type = 'test';

%% set parameters
Apad = Apad_map(layer_name);
Arf = Arf_map(layer_name);
Astride = Astride_map(layer_name);
offset = offset_map(layer_name); % This offset can fully get rid of out-boundry
% override offset to allow for some close-to-boundary patches
offset = 2; % Offset 2 is manually selected for pool4
feat_dim = featDim_map(layer_name);

% set image pathes
Dataset.img_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Images/%s_imagenet/';
Data.gt_dir = './intermediate/ground_truth_data/';
Dataset.train_list = fullfile(Data.gt_dir, ['%s_' sprintf('%s_train.txt', dataset_suffix)]);
Dataset.test_list =  fullfile(Data.gt_dir, ['%s_' sprintf('%s_test.txt', dataset_suffix)]);

dir_img = sprintf(Dataset.img_dir, category);
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


% load bounding boxes proposals
Data.root_dir = './intermediate/data/';
switch set_type
    case 'train'
        file_bbox_proposals = fullfile(Data.root_dir, sprintf('bbox_props_%s_%s_train.mat', category, dataset_suffix));
    case 'test'
        file_bbox_proposals = fullfile(Data.root_dir, sprintf('bbox_props_%s_%s_test.mat', category, dataset_suffix));
    otherwise
        error('Error: unknown set_type');
end  

assert( exist(file_bbox_proposals, 'file') > 0 );
load(file_bbox_proposals, 'Box');
assert(length(Box) == img_num);

% load VC dictionary
VC.dict_dir = './intermediate/dictionary/';
VC.layer = layer_name;
VC.num = 216;
file_VC_dict = fullfile(VC.dict_dir, sprintf('dictionary_imagenet_%s_vgg16_%s_K%i_norm_nowarp_prune_%i.mat', 'all', VC.layer, VC.num, feat_dim));
assert( exist(file_VC_dict, 'file') > 0 );
load(file_VC_dict, 'centers'); % 'centers' ~ [feat_dim, num_VC]
assert(size(centers, 1) == feat_dim);
assert(size(centers, 2) == VC.num);

%% initialize Caffe
Caffe.dir = '/media/zzs/5TB/tmp/caffe/';
addpath(fullfile(Caffe.dir, 'matlab'));

load('./ilsvrc_2012_mean.mat');
model = '/media/zzs/SSD1TB/zzs/surgeried/VGG_ILSVRC_16_layers_deploy_pool5.prototxt';
weights = '/media/zzs/SSD1TB/zzs/surgeried/surgery_weight';
mean_pixel = mean(mean(mean_data, 1), 2);
Caffe.gpu_id = 1;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(Caffe.gpu_id);
net = caffe.Net(model, weights, 'test');

%%
Data.root_dir2 = '/media/zzs/4TB/qingliu/qing_intermediate/';
dir_feat_bbox_proposals = fullfile(Data.root_dir2, 'feat');
MkdirIfMissing(dir_feat_bbox_proposals);

Feat.num_batch_img = 100;
Feat.max_num_props_per_img = 150;
num_batch = ceil(img_num / Feat.num_batch_img);

for i = 1: num_batch   
    file_cache_feat_batch = fullfile(dir_feat_bbox_proposals, sprintf('props_feat_%s_%s_%s_%d.mat', ...
                                      category, dataset_suffix, set_type, i));
    img_start_id = 1 + Feat.num_batch_img * (i - 1);
    img_end_id = min(Feat.num_batch_img * i, img_num);                              
                                  
    fprintf(' stack %d (%d ~ %d):', i, img_start_id, img_end_id);
    
    if exist(file_cache_feat_batch, 'file') > 0
        fprintf(' already found;\n');
        continue;
    end       
    
    feat = cell([1, img_end_id - img_start_id + 1]);    % feat{n} ~ struct('img_path', 'img_siz', 'box', 'box_siz', 'r_set')
    
    cnt_img = 0;
    for n = img_start_id: img_end_id
        cnt_img = cnt_img + 1;        
        
        file_img = sprintf('%s/%s.JPEG', dir_img, img_list{1}{n});
        img = imread(file_img);    
        [height, width, ~] = size(img);
        if size(img, 3) == 1
           img = repmat(img, [1 1 3]);
        end
        
        if strcmp(category, 'car')
            assert(Box(n).anno.height == height);
            assert(Box(n).anno.width == width);
            boxes = Box(n).boxes;
        else
            assert(strcmp(Box{n}.name, img_list{1}{n}));
            boxes = Box{n}.boxes;
        end
    
        boxes = boxes(1: min(Feat.max_num_props_per_img, size(boxes, 1)), :);
        num_box = size(boxes, 1);
    
        feat{cnt_img}.img_path = file_img;
        feat{cnt_img}.img_siz = [height, width];
        feat{cnt_img}.box = boxes(:, 1: 4);
        feat{cnt_img}.box_siz = zeros([num_box, 2]);
        feat{cnt_img}.r = cell([num_box, 1]);
        
        for j = 1: num_box
            bbox = boxes(j, 1: 4);
            bbox = [max(ceil(bbox(1)), 1), max(ceil(bbox(2)), 1), min(floor(bbox(3)), width), min(floor(bbox(4)), height)];
            
            % crop and resize image patch for 'bbox'
            patch = img(bbox(2): bbox(4), bbox(1): bbox(3), :);
            scaled_patch = myresize(patch, caffe_dim, 'short');
            
            feat{cnt_img}.box_siz(j, 1) = size(scaled_patch, 1);
            feat{cnt_img}.box_siz(j, 2) = size(scaled_patch, 2);
            
            % compute deep network layer features
            data = single(scaled_patch(:, :, [3, 2, 1]));
            data = bsxfun(@minus, data, mean_pixel);
            data = permute(data, [2, 1, 3]);
            net.blobs('data').reshape([size(data), 1]);
            net.reshape();

            net.forward({data});
            layer_feature = permute(net.blobs(layer_name).get_data(), [2, 1, 3]);
            
            % compute distance features ('r_set')
            h = size(layer_feature, 1);
            w = size(layer_feature, 2);
            layer_feature = reshape(layer_feature, [], feat_dim)';
            feat_norm = sqrt(sum(layer_feature.^2, 1));
            layer_feature = bsxfun(@rdivide, layer_feature, feat_norm);
            layer_feature = matrixDist(layer_feature, centers)';
            layer_feature = reshape(layer_feature, h, w, []);
            assert(size(layer_feature, 3) == VC.num);
            
            feat{cnt_img}.r{j} = layer_feature;            
        end % j: box index
        
        if mod(cnt_img, 10) == 0
            fprintf(' %d', n);
        end
        
    end % n: image index       
    
    save(file_cache_feat_batch, 'feat', '-v7.3');
    
    fprintf('\n');
    
end % i: stack index

end % end of function

