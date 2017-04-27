% compute and cache VC data

function comptVCData_bg(set_type, config_file)
try
    eval(config_file);
catchcat 
    keyboard;
end


switch set_type
    case 'train'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_train_bg.mat', category, dataset_suffix));
    case 'test'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_test_bg.mat', category, dataset_suffix));
    otherwise
        error('Error: unknown set_type;\n');
end


try
    load(file_cache_VC_data, 'r_set')
    assert(exist('r_set', 'var')>0)
catch
    fprintf('compute and cache VC distance data for "%s" set backgrounds ...\n', set_type);
    VC.num = 216;
    VC.layer = layer_name;
    
    feat_dim = featDim_map(VC.layer);
    
    %% load VC dictionary
    load(sprintf(Dictionary.new_dir, 'all2', layer_name, VC.num), 'centers');
    
    % 'centers' ~ [feat_dim, num_VCs]
    assert(size(centers, 1) == feat_dim);
    assert(size(centers, 2) == VC.num);
    
    switch set_type
        case 'train'
            file_cache_feat = fullfile(Feat.cache_dir, sprintf('%s_%s_train_bg.mat', category, dataset_suffix));
        case 'test'
            file_cache_feat = fullfile(Feat.cache_dir, sprintf('%s_%s_test_bg.mat', category, dataset_suffix));
        otherwise
            error('Error: unknown set_type;\n');
    end
    assert( exist(file_cache_feat, 'file') > 0 );
    load(file_cache_feat, 'feat_set');
    
    feat_set_cnt = 0;
    for n = 1: length(feat_set)
        if ~isempty(feat_set{n})
            feat_set_cnt = feat_set_cnt+1;
        end
    end
    
    fprintf('%d images have background features. \n', feat_set_cnt);
    
    r_set = cell(1, feat_set_cnt);
    
    r_set_i = 0;
    for n = 1: length(feat_set)                                                % for each image
        if isempty(feat_set{n})
            continue
        end
        r_set_i = r_set_i+1;
        
        layer_feature = feat_set{n};
        height = size(layer_feature, 1);
        width = size(layer_feature, 2);
        layer_feature = reshape(layer_feature, [], feat_dim)';
        feat_norm = sqrt(sum(layer_feature.^2, 1));
        layer_feature = bsxfun(@rdivide, layer_feature, feat_norm);
        layer_feature = matrixDist(layer_feature, centers)';
        layer_feature = reshape(layer_feature, height, width, []);
        assert(size(layer_feature,3)==size(centers,2));
        
        r_set{r_set_i} = layer_feature;
        
        if mod(n, 100) == 0
            disp(n);
        end
    end
    
    if exist(file_cache_VC_data, 'file')
        save(file_cache_VC_data, 'r_set', '-append');
    else
        save(file_cache_VC_data, 'r_set', '-v7.3');
    end
end

end % end of function