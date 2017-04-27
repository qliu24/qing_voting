% extract deep network layer features

function extractLayerFeat_bg(set_type, config_file)
try    
    eval(config_file);
catch
    keyboard;
end

fprintf('extract deep network layer features of background for "%s" set ...\n', set_type);

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

no_bg_num=0;
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
    total_bboxs = size(anno.objects,2);
    
    bbox = [max(ceil(bbox(1)), 1), max(ceil(bbox(2)), 1), min(floor(bbox(3)), width), min(floor(bbox(4)), height)];
    bbox_h = bbox(4)-bbox(2);
    bbox_w = bbox(3)-bbox(1);
    
    find_bg = zeros(1,total_bboxs);
    rdm_cnt = 0;
    while ~all(find_bg) & rdm_cnt < 10000
        rdm_cnt = rdm_cnt+1;
        find_bg = zeros(1,total_bboxs);
        r_h = randi(height-bbox_h);
        r_w = randi(width-bbox_w);
        bbox_bg = [r_w, r_h, r_w+bbox_w, r_h+bbox_h];
        
        for bbi = 1:total_bboxs
            bbox = anno.objects(bbi).bbox;
            bbox = [max(ceil(bbox(1)), 1), max(ceil(bbox(2)), 1), min(floor(bbox(3)), width), min(floor(bbox(4)), height)];
            bbox_area = (bbox(3)-bbox(1))*(bbox(4)-bbox(2));
            over_1 = max(bbox(1), bbox_bg(1));
            over_3 = min(bbox(3), bbox_bg(3));
            
            over_2 = max(bbox(2), bbox_bg(2));
            over_4 = min(bbox(4), bbox_bg(4));
            
            if over_1 >= over_3 | over_2 >= over_4
                find_bg(bbi) = true;
                continue
            end
            
            over_area = (over_3-over_1)*(over_4-over_2);
            if over_area/bbox_area < 0.6
                find_bg(bbi) = true;
                continue
            end
        end
    end
    
    if rdm_cnt==10000
        no_bg_num = no_bg_num+1;
        continue
    end
    
    patch = img(bbox_bg(2): bbox_bg(4), bbox_bg(1): bbox_bg(3), :);
    % imwrite(patch, sprintf('./test/test_bg_%d.png', n));
    
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

fprintf('%d out of %d images got no backgrounds. \n', no_bg_num, img_num)
caffe.reset_all;

MkdirIfMissing(Feat.cache_dir);
switch set_type
    case 'train'
        file_cache_feat = fullfile(Feat.cache_dir, sprintf('%s_%s_train_bg.mat', category, dataset_suffix));
    case 'test'
        file_cache_feat = fullfile(Feat.cache_dir, sprintf('%s_%s_test_bg.mat', category, dataset_suffix));
    otherwise
        error('Error: unknown type of dataset;\n');
end
save(file_cache_feat, 'feat_set', '-v7.3');

end % end of function