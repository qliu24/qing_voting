% extract deep network layer features

function extractLayerFeat(set_type, config_file)
try    
    eval(config_file);
catch
    keyboard;
end

fprintf('extract deep network layer features for "%s" set ...\n', set_type);

caffe.set_mode_gpu();
caffe.set_device(GPU_id);
net = caffe.Net(model, weights, 'test');

dir_img = sprintf(Dataset.img_dir, category);
dir_anno = sprintf(Dataset.anno_dir, category);

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

select_num = img_num;
feat_set = cell(1, select_num);
for n = 1: select_num                       % for each image   
    
    file_img = sprintf('%s/%s.JPEG', dir_img, img_list{1}{n});
    img = imread(file_img);
    [height, width, ~] = size(img);
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end

    file_anno = sprintf('%s/%s.mat', dir_anno, img_list{1}{n});
    anno = load(file_anno);
    anno = anno.record;
    bbox = anno.objects(img_list{2}(n)).bbox;
    
    bbox = [max(ceil(bbox(1)), 1), max(ceil(bbox(2)), 1), min(floor(bbox(3)), width), min(floor(bbox(4)), height)];
    patch = img(bbox(2): bbox(4), bbox(1): bbox(3), :);
    scaled_patch = myresize(patch, caffe_dim, 'short');
    
    % compute deep network features
    data = single(scaled_patch(:,:,[3, 2, 1]));
    data = bsxfun(@minus, data, mean_pixel);
    data = permute(data, [2, 1, 3]);
    net.blobs('data').reshape([size(data), 1]);
    net.reshape();
    net.forward({data});
    layer_feature = permute(net.blobs(layer_name).get_data(), [2, 1, 3]);
    feat_set{n} = layer_feature;  
    
    if mod(n,100) == 0
        disp(n);
    end    
end % image

%%
MkdirIfMissing(Feat.cache_dir);
switch set_type
    case 'train'
        file_cache_feat = fullfile(Feat.cache_dir, sprintf('%s_%s_train.mat', category, dataset_suffix));
    case 'test'
        file_cache_feat = fullfile(Feat.cache_dir, sprintf('%s_%s_test.mat', category, dataset_suffix));
    otherwise
        error('Error: unknown type of dataset;\n');
end
save(file_cache_feat, 'feat_set', '-v7.3');

end % end of function
