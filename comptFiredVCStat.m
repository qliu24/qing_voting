% compute the statistics of the number of fired VC patches
% created on 10/06/16, by Jun Zhu @JHU

function comptFiredVCStat(set_type, config_file)

%%
try    
    eval(config_file);
catch
    keyboard;
end

fprintf('compute the statistics of fired VC patches for "%s" set ...\n', set_type);

%%
% 


num_thh = length(Thresh.cand_values);

dir_img = sprintf(Dataset.img_dir, category);
dir_anno = sprintf(Dataset.anno_dir, category);

switch set_type
    case 'train'
        file_list = sprintf(Dataset.train_list, category);
        file_fgmask_viewpoint = sprintf(Dataset.mask_viewpoint_data, 'train');
    case 'test'
        file_list = sprintf(Dataset.test_list, category);
        file_fgmask_viewpoint = sprintf(Dataset.mask_viewpoint_data, 'test');
    otherwise
        error('Error: unknown set_type');
end

assert(exist(file_list, 'file') > 0);
file_ids = fopen(file_list, 'r');
img_list = textscan(file_ids, '%s %d');
num_img = length(img_list{1});

assert(exist(file_fgmask_viewpoint, 'file') > 0);
mask_vp = load(file_fgmask_viewpoint, 'data');
assert(length(mask_vp.data) == num_img);

% load distance data of VC patches
switch set_type
    case 'train'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_train.mat', category, dataset_suffix));
    case 'test'
        file_cache_VC_data = fullfile(VC.cache_dir, sprintf('%s_%s_test.mat', category, dataset_suffix));
    otherwise
        error('Error: unknown set_type;\n');
end

assert( exist(file_cache_VC_data, 'file') > 0 );
load(file_cache_VC_data, 'r_set');

assert( length(r_set) == num_img );
% assert( size(r_set{1}, 3) == K );
fprintf('K for current dictionary: %d\n', size(r_set{1}, 3))
K = size(r_set{1}, 3);

%%
tot_num_patches = zeros([num_img, 1], 'uint32');
hist_vc_patches = cell([num_img, 1]);
valid_seg_mask = true([num_img, 1]);

file_vc_patch_stat = fullfile(Feat.cache_dir, sprintf('vc_patch_stat_%s_%s_%s.mat', category, dataset_suffix, set_type));

try
    load(file_vc_patch_stat, 'tot_num_patches', 'hist_vc_patches', 'all_thh');
    
catch

for n = 1: num_img % for each image   
    
    img_name = img_list{1}{n};
    assert(strcmp(mask_vp.data{n}.img_name, img_name));

    file_img = sprintf('%s/%s.JPEG', dir_img, img_name);
    img = imread(file_img);
    [img_height, img_width, ~] = size(img);
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end

    obj_idx = img_list{2}(n);
    assert(mask_vp.data{n}.obj_idx == obj_idx);
    file_anno = sprintf('%s/%s.mat', dir_anno, img_name); 
    anno = load(file_anno);
    anno = anno.record;
    bbox = anno.objects(obj_idx).bbox;
    
    bbox = [max(ceil(bbox(1)), 1), max(ceil(bbox(2)), 1), min(floor(bbox(3)), img_width), min(floor(bbox(4)), img_height)];
    patch = img(bbox(2): bbox(4), bbox(1): bbox(3), :);
    scaled_patch = myresize(patch, caffe_dim, 'short');
    
    height = size(scaled_patch, 1);
    width = size(scaled_patch, 2);
    
    if isempty(mask_vp.data{n}.deeplab_mask)
        valid_seg_mask(n) = false;
        continue;
    end
    
    mask = (mask_vp.data{n}.deeplab_mask == 1);
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     try
%         assert(height == size(mask, 1));
%         assert(width == size(mask, 2));
%     catch
%         keyboard;
%     end
    
    if (size(mask, 1) ~= height) || (size(mask, 2) ~= width)
        mask = imresize(mask, [height, width]);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    r = r_set{n};
    num_rows_vc_patch = size(r, 1);
    num_cols_vc_patch = size(r, 2);
         
    i_y_vc_patch = 1: num_rows_vc_patch;
    y_vc_patch = Astride * (i_y_vc_patch - 1) + 1 - Apad + (Arf / 2);
%     y_vc_patch_ = y_vc_patch;
    i_y_vc_patch = i_y_vc_patch(y_vc_patch <= height);
    y_vc_patch = y_vc_patch(y_vc_patch <= height);
    
    i_x_vc_patch = 1: num_cols_vc_patch;
    x_vc_patch = Astride * (i_x_vc_patch - 1) + 1 - Apad + (Arf / 2);
%     x_vc_patch_ = x_vc_patch;
    i_x_vc_patch = i_x_vc_patch(x_vc_patch <= width);
    x_vc_patch = x_vc_patch(x_vc_patch <= width);
    
    [i_X_vc_patch, i_Y_vc_patch] = meshgrid(i_x_vc_patch, i_y_vc_patch);
    [X_vc_patch, Y_vc_patch] = meshgrid(x_vc_patch, y_vc_patch);
%     [X_vc_patch_, Y_vc_patch_] = meshgrid(x_vc_patch_, y_vc_patch_);
       
    try
        L_vc_patch = sub2ind([height, width], Y_vc_patch(:), X_vc_patch(:));
        i_L_vc_patch = sub2ind([num_rows_vc_patch, num_cols_vc_patch], i_Y_vc_patch(:), i_X_vc_patch(:));
        assert( length(L_vc_patch) == length(i_L_vc_patch) );
    catch
        keyboard;
    end
    ind_fg_vc_patch = find(mask(L_vc_patch));    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     visFgVcPatches(img, bbox, scaled_patch, mask, L_vc_patch, ind_fg_vc_patch);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    if isempty(ind_fg_vc_patch)
        valid_seg_mask(n) = false;
        continue;
    end
    
    i_L_vc_patch = i_L_vc_patch(ind_fg_vc_patch);
    L_vc_patch = L_vc_patch(ind_fg_vc_patch);

    tot_num_patches(n) = length(ind_fg_vc_patch);   
    d = false([tot_num_patches(n), K, num_thh]);
    
    %
    assert( size(r, 1) == num_rows_vc_patch );
    assert( size(r, 2) == num_cols_vc_patch );
    r = reshape(r, [num_rows_vc_patch * num_cols_vc_patch, K]);
    r = r(i_L_vc_patch, :);                                             % ~ [num_fg_patches, num_VC]
    
    for j = 1: num_thh
        thh = Thresh.cand_values(j);
        d(:, :, j) = (r <= thh);
    end % j
    
    hist_vc_patches{n} = d;
    
    if mod(n, 50) == 0
        fprintf(' %d', n);
    end
    
end % n
fprintf('\n');

tot_num_patches = tot_num_patches(valid_seg_mask);
hist_vc_patches = hist_vc_patches(valid_seg_mask);
all_thh = Thresh.cand_values;

%%
save(file_vc_patch_stat, 'tot_num_patches', 'hist_vc_patches', 'valid_seg_mask', 'all_thh');

end

%% draw curves
all_thh = all_thh(:);
num_thh = length(all_thh);
ave_num_fire_vc_patches = zeros([num_thh 5]);    % '#vc = 0', '#vc = 1', '#vc = 2', '#vc = 3', '#vc > 3'

for i = 1: length(hist_vc_patches)
    d = hist_vc_patches{i};
    
    d = sum(d, 2);
    d = permute(d, [3 1 2]);
    
    % '#vc = 0'
    ave_num_fire_vc_patches(:, 1) = ave_num_fire_vc_patches(:, 1) + sum(d == 0, 2);
    % '#vc = 1'
    ave_num_fire_vc_patches(:, 2) = ave_num_fire_vc_patches(:, 2) + sum(d == 1, 2);
    % '#vc = 2'
    ave_num_fire_vc_patches(:, 3) = ave_num_fire_vc_patches(:, 3) + sum(d == 2, 2);
    % '#vc = 3'
    ave_num_fire_vc_patches(:, 4) = ave_num_fire_vc_patches(:, 4) + sum(d == 3, 2);
    % '#vc > 3'
    ave_num_fire_vc_patches(:, 5) = ave_num_fire_vc_patches(:, 5) + sum(d > 3, 2);
end % i

ave_num_fire_vc_patches = ave_num_fire_vc_patches / sum(tot_num_patches);

figure;
plot(all_thh, ave_num_fire_vc_patches, 'LineWidth', 2);
xlabel('threshold value');
ylabel('average number of VC patches');
axis([0 1.2 0 1])
legend({'#vc = 0', '#vc = 1', '#vc = 2', '#vc = 3', '#vc > 3'});

fig_vc_patch_stat = fullfile(Feat.cache_dir, sprintf('vc_patch_stat_%s_%s_%s.fig', category, dataset_suffix, set_type));
saveas(gcf, fig_vc_patch_stat);
end % end of function
