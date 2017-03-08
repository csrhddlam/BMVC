%% init
input_7_7_176 = rand(7, 7, 176) + 0.645; % input vector containing distances to VCs
load('matrix_23'); % load model params variable, 'matrix'
visible = 7 * 7 * 176; hidden = 1; % constants

%% interface
tic;
score = interface_zhishuai(input_7_7_176, matrix, visible, hidden);
elapsed_time = toc;

%% print
disp(['Score_of_patch: ', num2str(score)]);
disp(['Time_consumption: ', num2str(elapsed_time), ' seconds']);