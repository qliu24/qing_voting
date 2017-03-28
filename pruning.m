function pruning(config_file)
% first load dictionary and original cluster feature set.
try
    eval(config_file)
catch
    keyboard
end

try
    assert(exist(sprintf(Dictionary.new_dir, category, layer_name, VC.num), 'file')>0);
catch
    
    load(sprintf(Dictionary.original_dir,category, layer_name, cluster_num));
    load(sprintf(Dictionary.feature_cache_dir,category, layer_name));
    fprintf('pruning the clusters \n')
    
    %% decide the rank of clusters
    [~, K] = size(centers);
    % L2 normalization
    feat_norm = sqrt(sum(feat_set_all.^2, 1));
    feat_set_all = bsxfun(@rdivide, feat_set_all, feat_norm);
    count = hist(double(assignment), K);
    
    % based on centers
    pw_cen = zeros(K, K);
    for k  = 1:K
        for m = 1:K
            pw_cen(k,m) = norm(centers(:, k)-centers(:, m));
        end
    end
    
    % based on data points
    pw_all = zeros(K, K);
    for k = 1:K
        target = centers(:, k);
        for m = 1:K
            index = find(assignment==m);
            temp_feat = feat_set_all(:, index);
            dist = sqrt(sum(bsxfun(@minus, target, temp_feat).^2, 1));
            [sort_value, sort_idx] = sort(dist, 'ascend');
            pw_all(k, m) = mean(sort_value(1:round(0.95*length(sort_value))));
        end
        %         k
    end
    
    % compute metric
    list = zeros(1, K);
    for k = 1:K
        rec = zeros(1, K);
        for m = 1:K
            if m ~= k
                rec(m) = (pw_all(m, m) + pw_all(k, k))/pw_cen(m, k);
            end
        end
        list(k) = max(rec);
    end
    
    % the lower the better
    [aaa, bbb] = sort(list, 'ascend');
    sort_list = [aaa; bbb];
    % the higher the better
    count_norm = count/sum(count);
    [aaa, bbb] = sort(count_norm, 'descend');
    sort_count_norm = [aaa; bbb];
    % give big penalty if cluster number is too small
    penalty = 100*(count<100);
    % combine the above metrics, the lower the better
    com = list - K*count_norm + penalty;
    [aaa, bbb] = sort(com, 'ascend');
    sort_com = [aaa; bbb];
    
    %% greedy pruning
    fprintf('greedy pruning... \n')
    sort_cls = sort_com(2, :);
    rec = ones(1, K);
    thresh1 = 0.95;
    thresh2 = 0.2;
    prune = [];
    prune_res = [];
    
    while sum(rec) > 0
        temp = [];
        idx = find(rec==1, 1);
        cls = sort_cls(idx);
        target = centers(:, cls);
        index = find(assignment==cls);
        temp_feat = feat_set_all(:, index);
        dist = sqrt(sum(bsxfun(@minus, target, temp_feat).^2, 1));
        [sort_value, sort_idx] = sort(dist, 'ascend');
        dist_thresh = sort_value(round(thresh1*length(sort_value)));
        rec(idx) = 0;
        for n = idx+1:K
            if rec(n) == 1
                index = find(assignment==sort_cls(n));
                temp_feat = feat_set_all(:, index);
                dist = sqrt(sum(bsxfun(@minus, target, temp_feat).^2, 1));
                if mean(dist<dist_thresh) >= thresh2
                    temp = [temp, [n; sort_cls(n); mean(dist<dist_thresh)]];
                    rec(n) = 0;
                end
            end
        end
        fprintf('%d, %d, %d\n', idx, cls, size(temp, 2));
        prune = [prune, [idx; cls; size(temp, 2)]];
        prune_res = [prune_res, {temp}];
    end
    pruning_table=cell(size(prune, 2));
    
    
    
    %% update new dictionary
    fprintf('update new dictionary... \n')
    K_new = size(prune, 2);
    centers_new = zeros(size(centers,1), K_new);
    assignment_new = 0*assignment;
    for k = 1:K_new
        if isempty(prune_res{k})
            temp = prune(2, k);
        else
            temp = [prune(2, k), prune_res{k}(2, :)];
        end
        pruning_table{k}=temp;
        weight = count(1, temp);
        weight = weight/sum(weight);
        temp_cen = centers(:, temp);
        centers_new(:, k) = temp_cen*weight';
        for i = 1:length(temp)
            assignment_new(assignment==temp(i)) = k;
        end
    end
    
    
    
    
    % save as top 100 typical examples for each cluster
    K = K_new;
    centers = centers_new;
    assignment = assignment_new;
    num = 100;
    example = cell(1, K);
    padSize = 100;
    patch_size = loc_set_all(4, 1) - loc_set_all(2, 1);
    fprintf('save TOP %d images for the new cluster \n', num);
    for k = 1:K
        target = centers(:, k);
        index = find(assignment == k);
        tempFeat = feat_set_all(:,index);
        error = sum(bsxfun(@minus, target, tempFeat).^2,1);
        [sort_value, sort_idx] = sort(error, 'ascend');
        % num=size(index, 2);
        patch_set = zeros(patch_size^2*3, num, 'uint8');
        for idx = 1:min(num, size(index, 2))
            img_id = loc_set_all(1, index(sort_idx(idx)));
            img = img_set_all{img_id};
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
    
    save(sprintf(Dictionary.new_dir,category, layer_name, K), 'assignment', 'centers', 'example', '-v7.3');
end
end

