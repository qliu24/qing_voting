function cluster(config_file)
% This script does kmeans clustering

try
    eval(config_file)
catch
    keyboard
end

addpath('/media/zzs/SSD1TB/zzs/library/vlfeat/toolbox/');
vl_setup();

save_path = sprintf(Dictionary.original_dir, category, layer_name, cluster_num);
try
    assert(exist(save_path, 'file')>0)
catch
    load(sprintf(Dictionary.feature_cache_dir, category, layer_name));

    % L2 normalization as preprocessing
    feat_norm = sqrt(sum(feat_set_all.^2, 1)); %#ok<*NODEF>
    feat_set_all = bsxfun(@rdivide, feat_set_all, feat_norm);

    [centers, assignment] = vl_kmeans(feat_set_all, cluster_num, 'Initialization', 'plusplus');
    % figure, hist(double(assignment), cluster_num);
    save(save_path, 'assignment', 'centers', '-v7.3');

    num = 100;  %the num of images for each cluster

    fprintf('save top %d images for each cluster', num);
    % save as top 100 typical examples for each cluster
    example = cell(1, cluster_num);
    % this padSize should be large enough, the value itself does not matter
    padSize = 100;
    patch_size = loc_set_all(4, 1) - loc_set_all(2, 1);
    for k = 1:cluster_num
        target = centers(:, k);
        index = find(assignment == k);
        tempFeat = feat_set_all(:,index);
        error = sum(bsxfun(@minus, target, tempFeat).^2,1);
        [~, sort_idx] = sort(error, 'ascend');
        patch_set = zeros(patch_size^2*3, num, 'uint8');
        for idx = 1:min(num,sum(assignment==k))
            img_id = loc_set_all(1, index(sort_idx(idx)));
            img = img_set_all{img_id}; %#ok<*USENS>
            padPatch = padarray(img, [padSize, padSize]);
            col1 = loc_set_all(2, index(sort_idx(idx))) + padSize;
            row1 = loc_set_all(3, index(sort_idx(idx))) + padSize;
            col2 = loc_set_all(4, index(sort_idx(idx))) + padSize;
            row2 = loc_set_all(5, index(sort_idx(idx))) + padSize;
            patch = padPatch(row1+1:row2, col1+1:col2, :);
            patch_set(:, idx) = patch(:);
        end
        example{k} = patch_set;
        if mod(k, 20) == 0
            fprintf('  %d', k);
        end
    end

    save(save_path, 'assignment', 'centers', 'example', '-v7.3');
end
end
