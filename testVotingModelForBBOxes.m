% test voting models for bounding box proposals 
% created by Jun Zhu @JHU, on 11/11/2016.

function testVotingModelForBBOxes(config)
try
    eval(config)
catch
    keyboard
end
%%
fprintf('test voting models for bounding box proposals on "%s" set ...\n', set_type);

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

%%
% load model
load(Model_file);

% weight for unary models
% mixture_weights, mixture_priors for mixture models
% 
if strcmp(model_type, 'single')
    weight_obj = permute(weight,[3,2,1]);
elseif strcmp(model_type, 'mix')
    weight_objs = cell(size(mixture_weights,1),1);
    log_priors = cell(size(mixture_weights,1),1);
    for mm=1:size(mixture_weights,1)
        weight_objs{mm} = reshape(mixture_weights(mm,:), temp_dim(model_category));
        log_priors{mm} = log(mixture_priors(mm));
    end
else
    error('Error: unknown model_type');
end

% load(Model_file_bg);
% weight_obj_bg = permute(weight,[3,2,1]);

%% compute voting scores for each image

num_batch = length(dir(fullfile(dir_feat_bbox_proposals, ...
                                  sprintf('props_feat_%s_%s_%s_*.mat', category, dataset_suffix, set_type))));
                              
det = cell([img_num, 1]);   % det{n} ~ struct('img_path', 'img_siz', 'box', 'box_siz', 'score')                              
n = 0;             
fprintf('compute voting scores ...');
for i = 1: num_batch
    fprintf(' for batch %d of %d:', i, num_batch);
    
    file_cache_feat_batch = fullfile(dir_feat_bbox_proposals, sprintf('props_feat_%s_%s_%s_%d.mat', ...
                                      category, dataset_suffix, set_type, i));
    assert( exist(file_cache_feat_batch, 'file') > 0 );
    
    clear('feat');
    load(file_cache_feat_batch, 'feat');
            
    for cnt_img = 1: length(feat)
        n = n + 1;
        
        det{n}.img_path = feat{cnt_img}.img_path;
        det{n}.img_siz = feat{cnt_img}.img_siz;
        det{n}.box = feat{cnt_img}.box;
        det{n}.box_siz = feat{cnt_img}.box_siz;
                
        num_box = size(feat{cnt_img}.box, 1);        
        det{n}.score = zeros([num_box, 1]);
        for j = 1: num_box
            if strcmp(model_type, 'single')
                det{n}.score(j, 1) = comptScores(feat{cnt_img}.r{j}, weight_obj);
            elseif strcmp(model_type, 'mix')
                det{n}.score(j, 1) = comptScoresM(feat{cnt_img}.r{j}, weight_objs, log_priors);
            else
                error('Error: unknown model_type');
            end                
        end
        
        if mod(cnt_img, 10) == 0
            fprintf(' %d', n);
        end
    end % n: image index in batch
    
    fprintf('\n');
    
end % i: batch index
assert(n == img_num);

%%

save(file_det_result, 'det', '-v7.3');

end % end of function

function score = comptScoresM(input, weight_objs, log_priors)
    score_i = zeros(length(weight_objs),1);
    for mm=1:length(weight_objs)
        logllk = comptScores(input, weight_objs{mm});
        score_i(mm) = logllk+log_priors{mm};
    end
    score = logsumexp(score_i);
end
