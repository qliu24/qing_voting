% extract layer features for bounding-box proposals
function comptFeatForBBoxes(config)
try
    eval(config)
catch
    keyboard
end

%% load data
fprintf('extract deep network layer features for bounding box proposals on "%s" set ...\n', set_type);
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
assert( exist(file_VC_dict, 'file') > 0 );
load(file_VC_dict, 'centers'); % 'centers' ~ [feat_dim, num_VC]
assert(size(centers, 1) == feat_dim);
assert(size(centers, 2) == VC.num);

%% initialize Caffe
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(Caffe.gpu_id);
net = caffe.Net(model, weights, 'test');

%%
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