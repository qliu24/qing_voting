% merge the features from different objects

function merge_all_cat_dict(config_file, samp_size)

try
    eval(config_file)
catch
    keyboard
end

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};
save_dir = '/media/zzs/4TB/qing_intermediate/dictionary_imagenet_%s_vgg16_%s_nowarp.mat';
save_path = sprintf(save_dir, 'all', layer_name);

img_set_all = cell(1,0);
feat_set_all = single.empty(featDim,0);
loc_set_all = single.empty(5,0);
cnt_img=0;
for i = 1:numel(object)
    category = object{i}; % set the object of interest
    eval(config_file);
    fprintf(' %s', category)
    
    load(sprintf(Dictionary.feature_cache_dir, category, layer_name));
    % roughly balance the training example number across different category
    if length(img_set) > 1000
        img_set = img_set(1:1000);
        feat_set = feat_set(:,1:1000*samp_size);
        loc_set = loc_set(:, 1:1000*samp_size);
    end
    
    img_set_all = [img_set_all, img_set];
    feat_set_all = cat(2, feat_set_all, feat_set);
    
    loc_set(1,:) = loc_set(1,:)+cnt_img;
    cnt_img = max(loc_set(1,:));
    loc_set_all = cat(2, loc_set_all, loc_set);
    
    assert(cnt_img == length(img_set_all))
end

save(save_path, 'feat_set_all', 'img_set_all', 'loc_set_all', '-v7.3');

end % end of function
