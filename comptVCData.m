% compute and cache VC data
% 'zj_v1': created on 16/05/06, by Jun Zhu @JHU;

function comptVCData_zj_v1(set_type, config_file)
try
    eval(config_file);
catch
    keyboard;
end


switch set_type
    case 'train'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_train.mat', category, dataset_suffix));
    case 'test'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_test.mat', category, dataset_suffix));
    otherwise
        error('Error: unknown set_type;\n');
end


try
    load(file_cache_VC_data, 'r_set')
    assert(exist('r_set', 'var')>0)
catch
    fprintf('compute and cache VC distance data for "%s" set ...\n', set_type);
    VC.num = 216
    VC.layer = layer_name
    
    feat_dim = featDim_map(VC.layer);
    
    %% load VC dictionary
    load(sprintf(Dictionary.new_dir, 'all', layer_name, VC.num), 'centers');
    
    % 'centers' ~ [feat_dim, num_VCs]
    assert(size(centers, 1) == feat_dim);
    assert(size(centers, 2) == VC.num);
    
    switch set_type
        case 'train'
            file_cache_feat = fullfile(Feat.cache_dir, sprintf('%s_%s_train.mat', category, dataset_suffix));
        case 'test'
            file_cache_feat = fullfile(Feat.cache_dir, sprintf('%s_%s_test.mat', category, dataset_suffix));
        otherwise
            error('Error: unknown set_type;\n');
    end
    assert( exist(file_cache_feat, 'file') > 0 );
    load(file_cache_feat, 'feat_set');
    
    r_set = cell(size(feat_set));
    
    for n = 1: length(feat_set)                                                % for each image
        layer_feature = feat_set{n};
        height = size(layer_feature, 1);
        width = size(layer_feature, 2);
        layer_feature = reshape(layer_feature, [], feat_dim)';
        feat_norm = sqrt(sum(layer_feature.^2, 1));
        layer_feature = bsxfun(@rdivide, layer_feature, feat_norm);
        layer_feature = matrixDist(layer_feature, centers)';
        layer_feature = reshape(layer_feature, height, width, []);
        assert(size(layer_feature,3)==size(centers,2));
        
        r_set{n} = layer_feature;
        
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
