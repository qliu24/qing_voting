% Evaluate voting model for object detection

function evalVotingModelForObjectDetection(set_type)
%%
fprintf('Evaluate voting models for object detection task on "%s" set ...\n', set_type);

%% Caffe parameter
caffe_dim = 224; % caffe input dimension in deploy protobuf
layer_set = {'pool1', 'pool2', 'pool3', 'pool4', 'pool5'};
% in original image input space
Apad_set = [2, 6, 18, 42, 90]; % padding size
Astride_set = [2, 4, 8, 16, 32]; % stride size
featDim_set = [64, 128, 256, 512, 512]; % feature dimension
Arf_set = [6, 16, 44, 100, 212];
offset_set = ceil(Apad_set./Astride_set);

Apad_map = containers.Map(layer_set, Apad_set);
Arf_map = containers.Map(layer_set, Arf_set);
Astride_map = containers.Map(layer_set, Astride_set);
featDim_map = containers.Map(layer_set, featDim_set);
offset_map = containers.Map(layer_set, offset_set);

dataset_suffix = 'mergelist_rand';
layer_name = 'pool4';
category = 'car';
%%
Apad = Apad_map(layer_name);
Arf = Arf_map(layer_name);
Astride = Astride_map(layer_name);

%% set parameters and directories
% set image pathes
Dataset.img_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Images/%s_imagenet/';
Dataset.anno_dir = '/media/zzs/SSD1TB/zzs/dataset/PASCAL3D+_release1.1/Annotations/%s_imagenet/';
Data.gt_dir = './intermediate/ground_truth_data/';
Dataset.train_list = fullfile(Data.gt_dir, ['%s_' sprintf('%s_train.txt', dataset_suffix)]);
Dataset.test_list =  fullfile(Data.gt_dir, ['%s_' sprintf('%s_test.txt', dataset_suffix)]);

% read image list
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

dir_img = sprintf(Dataset.img_dir, category);
dir_obj_anno = sprintf(Dataset.anno_dir, category);

% performance evaluation
Data.root_dir2 = '/media/zzs/4TB/qingliu/qing_intermediate/';
dir_perf_eval = fullfile(Data.root_dir2, 'eval_obj_det');
MkdirIfMissing(dir_perf_eval);

file_gt_obj_anno = fullfile(dir_perf_eval, sprintf('gt_anno_%s_%s_%s', category, dataset_suffix, set_type));
file_perf_eval = fullfile(dir_perf_eval, sprintf('eval_%s_%s_%s.mat', ...
                                                  category, dataset_suffix, set_type));
                                                 
%% process ground-truth annotations
try
     load(file_gt_obj_anno, 'gt', 'n_pos');
catch
    % collect groundtruth object annotations from all images
    fprintf(' extract ground truth objects ...\n');
    
    n_pos = 0;
    gt(img_num) = struct('bbox', [], 'diff', [], 'det', []);
    for n = 1: img_num
        img_name = img_list{1}{n};
        
        file_img = sprintf('%s/%s.JPEG', dir_img, img_name);
        img = imread(file_img);
        [hgt_img, wid_img, ~] = size(img);
        
        file_obj_anno = sprintf('%s/%s.mat', dir_obj_anno, img_name);
        assert(exist(file_obj_anno, 'file') > 0);
        
        anno = load(file_obj_anno);
        anno = anno.record;      
        assert(anno.imgsize(2) == hgt_img);
        assert(anno.imgsize(1) == wid_img);
        
        for j = 1: length(anno.objects)
            if strcmp(anno.objects(j).class, category)        
                  gt_bbox_cls = anno.objects(j).bbox;
                  gt_bbox_cls(1) = max(ceil(gt_bbox_cls(1)), 1);            % x_min
                  gt_bbox_cls(2) = max(ceil(gt_bbox_cls(2)), 1);            % y_min
                  gt_bbox_cls(3) = min(floor(gt_bbox_cls(3)), wid_img);     % x_max
                  gt_bbox_cls(4) = min(floor(gt_bbox_cls(4)), hgt_img);     % y_max
        
                  gt(n).bbox = cat(2, gt(n).bbox, gt_bbox_cls');                % ~ [4 num_bbox]
                  gt(n).diff = cat(1, gt(n).diff, anno.objects(j).difficult);           % ~ [num_bbox 1]
            end
        end
        gt(n).det = false([size(gt(n).bbox, 2), 1]);                          % ~ [num_bbox 1]
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if isempty(gt(n).bbox)
            keyboard;
        end
        
        n_pos = n_pos + sum(~gt(n).diff);
    end % n

    save(file_gt_obj_anno, 'gt', 'n_pos', '-v7.3');
end

% % hash image ids
% hash=VOChash_init(gtids);
% 
% npos=0;
% gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
% for i=1:length(gtids)
%     % extract objects of class
%     clsinds = strmatch(cls,{recs(i).objects(:).class},'exact');
%     gt(i).BB = cat(1,recs(i).objects(clsinds).bbox)';
%     gt(i).diff = [recs(i).objects(clsinds).difficult];
%     gt(i).det = false(length(clsinds),1);
%     npos = npos + sum(~gt(i).diff);
% end

%% compute detection scores 
try
    load(file_perf_eval, 'boxes', 'scores', 'vp_labels', 'img_ids');
    
catch

fprintf(' compute scores and do NMS for detection result:');

% load detection result
file_det_result = fullfile(Data.root_dir2, 'result', sprintf('props_det_%s_%s_%s.mat', category, dataset_suffix, set_type));
assert( exist(file_det_result, 'file') > 0 );
load(file_det_result, 'det');

assert(length(det) == img_num);
assert(length(gt) == img_num);

boxes = cell([img_num 1]);              % ~ boxes{n}: ~ [num_bbox 4]
scores = cell([img_num 1]);             % ~ scores{n}: ~ [num_bbox 1]
img_ids = cell([img_num 1]);            % ~ img_ids{n}: ~ [num_bbox 1]


Eval.nms_bbox_ratio = 0.3;
for n = 1: img_num    
    % compute scores for proposal bounding boxes
    
    num_bbox = size(det{n}.score, 1);
    scores{n} = det{n}.score;
    boxes{n} = det{n}.box;
    
    % do NMS
    nms_list = nms([boxes{n}, scores{n}], Eval.nms_bbox_ratio);
    boxes{n} = boxes{n}(nms_list, :);
    scores{n} = scores{n}(nms_list);
    img_ids{n} = n * ones([length(nms_list) 1]);
    
%     valid_score_box = find(scores{n} > Eval.score_thresh);
%     if ~isempty(valid_score_box)
%         scores{n} = scores{n}(valid_score_box);
%         boxes{n} = det{n}.box(valid_score_box, :);
%         vp_labels{n} = vp_labels{n}(valid_score_box);
%     end
    
    if mod(n, 50) == 0
        fprintf(' %d', n);
    end
end % n: image index
fprintf('\n');

save(file_perf_eval, 'boxes', 'scores', 'img_ids', '-v7.3');

end

%% evaluate detection performance
fprintf(' evaluate P-R curve and AP ...\n');

Eval.overlap_type = 'iou';
Eval.ov_thresh = 0.5;

% sort detections by decreasing confidence
scores = cell2mat(scores);    % ~ [num_bbox_tot 1]
boxes = cell2mat(boxes)';     % ~ [4 num_bbox_tot]
img_ids = cell2mat(img_ids)'; % ~ [num_bbox_tot 1]

[~, si] = sort(-scores);
img_ids = img_ids(si);
boxes = boxes(:, si);

% assign detections to ground truth objects
nd = length(scores);
tp = zeros([nd, 1]);
fp = zeros([nd, 1]);
tic;
for d = 1: nd
    % display progress
    if toc > 1
        fprintf('%s: pr: compute: %d|%d\n', category, d, nd);
        drawnow;
        tic;
    end
    
    % find ground truth image
    i = img_ids(d);   % i: image id

    % assign detection to ground truth object if any
    bb = boxes(:, d);
    ovmax = -inf;
    jmax = [];
    for j = 1: size(gt(i).bbox, 2)
        bbgt = gt(i).bbox(:, j);
        bi = [max(bb(1), bbgt(1)); max(bb(2), bbgt(2)); min(bb(3), bbgt(3)); min(bb(4), bbgt(4))];   % intersection
        iw = bi(3) - bi(1) + 1;
        ih = bi(4) - bi(2) + 1;
        if (iw > 0) && (ih > 0)                
            % compute overlap as area of intersection / area of union
            ua = (bb(3) - bb(1) + 1) * (bb(4) - bb(2) + 1) + ...
                 (bbgt(3) - bbgt(1) + 1) * (bbgt(4) - bbgt(2) + 1) - ...
                  iw * ih;    % area of union
            ov = iw * ih / ua;
            if ov > ovmax
                ovmax = ov;
                jmax = j;
            end
        end
    end
    
    % assign detection as true positive/don't care/false positive
    if ovmax >= Eval.ov_thresh
        if ~gt(i).diff(jmax)
            if ~gt(i).det(jmax)
                tp(d) = 1;            % true positive
                gt(i).det(jmax) = true;
            else
                fp(d) = 1;            % false positive (multiple/duplicate detection)
            end
        end
    else
        fp(d)=1;                    % false positive
    end
    
end % d: detection

% compute precision/recall
fp = cumsum(fp);
tp = cumsum(tp);
rec = tp / n_pos;
prec = tp ./ (fp + tp);

ap = VOCap(rec, prec);
fprintf(' AP = %2.1f', 100 * ap);

save(file_perf_eval, 'fp', 'tp', 'rec', 'prec', 'ap', '-append');

Eval.vis_prc = true;
if Eval.vis_prc
    % plot precision/recall
    plot(rec, prec, '-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, set: %s_%s, AP = %2.1f', ...
           category, dataset_suffix, set_type, 100*ap));
end

end % end of function

