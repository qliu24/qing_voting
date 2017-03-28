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
layer_name = 'pool4';


fprintf('cluster');
cluster(config);

