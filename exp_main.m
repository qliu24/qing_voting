%% this is the new script written by cihang, which includes all the steps that needed for the voting scheme
% --------- zhishuai@JHU 12/AUG

% the train & test list are obtained based on current available gt data
%%
clear
close all

global category layer_name GPU_id

object = {'car', 'aeroplane', 'bicycle', 'bus', 'motorbike', 'train'};

config = 'config_voting';
GPU_id = 0;


%% script begins
for i = 1:numel(object)
    category = object{i}; % set the object of interest
    layer_name = 'pool4'; % set the layer of interest

    try
        eval(config);
    catch
        keyboard;
    end

    %% -------------- data preparation part -------------------------------

    %% from training and testing dataset, we want to get Visual Concept
    % first step is to extract features from images at a certain layer
    fprintf('dictionary_nowarp');
    dictionary_nowarp(config);

end
