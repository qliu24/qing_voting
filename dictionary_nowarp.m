function dictionary_nowarp(config_file, samp_size)
% This script extracts cnn features and patches for kmeans clustering
% instead of warping to 224*224 which destroys the aspect ratio
% I resize the short size to 224 while keeping the aspect ratio

try
    eval(config_file)
catch
    keyboard
end

% training dataset
imgDir = sprintf(Dataset.img_dir, category);
annoDir = sprintf(Dataset.anno_dir, category);

fileName = sprintf(Dataset.train_list, category);
assert(exist(fileName, 'file')>0);
fileID = fopen(fileName, 'r');
imgList1 = textscan(fileID, '%s %d');


fileName = sprintf(Dataset.test_list, category);
assert(exist(fileName, 'file')>0);
fileID = fopen(fileName, 'r');
imgList2 = textscan(fileID, '%s %d');

imgList = cell(1,2)
imgList{1} = [imgList1{1}; imgList2{1}]
imgList{2} = [imgList1{2}; imgList2{2}]
imgNum = length(imgList{1});

save_path = sprintf(Dictionary.feature_cache_dir, category, layer_name);

try
    assert(exist(save_path, 'file')>0)
catch
    caffe.set_mode_gpu();
    caffe.set_device(GPU_id);
    net = caffe.Net(model, weights, 'test');
    [path, ~, ~] = fileparts(save_path);
    mkdir_if_missing(path);
    fprintf('extract deep network features on %s layer to form dictionary\n', layer_name);
    feat_set = zeros(featDim, samp_size*imgNum, 'single');
    loc_set = zeros(5, samp_size*imgNum, 'single');
    img_set = cell(1, imgNum);
    cnt = 1;
    
    for n = 1:imgNum
        imgPath = sprintf('%s/%s.JPEG', imgDir, imgList{1}{n});
        annoPath = sprintf('%s/%s.mat', annoDir, imgList{1}{n});
        img = imread(imgPath);
        [height, width, ~] = size(img);
        if size(img, 3) == 1
            img = cat(3, img, img, img);
        end
        anno = load(annoPath);
        anno = anno.record;
        bbox = anno.objects(imgList{2}(n)).bbox;
        bbox = [max(ceil(bbox(1)), 1), max(ceil(bbox(2)), 1), min(floor(bbox(3)), width), min(floor(bbox(4)), height)];
        patch = img(bbox(2):bbox(4), bbox(1):bbox(3), :);
        scalePatch = myresize(patch, caffe_dim, 'short');
        
        data = single(scalePatch(:,:,[3, 2, 1]));
        data = bsxfun(@minus, data, mean_pixel);
        data = permute(data, [2, 1, 3]);
        
        net.blobs('data').reshape([size(data), 1]);
        net.reshape();
        
        net.forward({data});
        layer_feature = permute(net.blobs(layer_name).get_data(), [2, 1, 3]);
        
        height = floor((size(scalePatch, 1) + 2*Apad - Arf)/Astride + 1);
        width = floor((size(scalePatch, 2) + 2*Apad - Arf)/Astride + 1);
        % we have to start from offset since some close-to-boundary positions on
        % feature map do not correspond to patches in the input image space due
        % to padding effects
        if strcmp(layer_name,'pool5')==1  % for pool5 layer, we do not have enough samples
            samp_size=(height-2*offset)*(width-2*offset);
        end
        if (height-2*offset)*(width-2*offset) >= samp_size
            samp_list = randperm((height-2*offset)*(width-2*offset), samp_size);
        else
            samp_list = randi((height-2*offset)*(width-2*offset), 1, samp_size);
        end
        [rlist, clist] = ind2sub([height-2*offset, width-2*offset], samp_list);
        for idx = 1:samp_size
            row = Astride*(rlist(idx)+offset-1)+1-Apad;
            col = Astride*(clist(idx)+offset-1)+1-Apad;
            feat_set(:, cnt) = squeeze(layer_feature(rlist(idx)+offset, clist(idx)+offset, :));
            loc_set(:, cnt) = [n; col; row; col+Arf; row+Arf];
            cnt = cnt + 1;
        end
        img_set{n} = scalePatch;
        if mod(n,20) == 0
            fprintf(' %d', n);
        end
    end
    save(save_path, 'feat_set', 'img_set', 'loc_set', '-v7.3');
end


end