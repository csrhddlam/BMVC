%% constants
gpuDevice(1)
learning_rate = 1e1;
momentum = 0.1;
weight_decay = 0.0001;

label_val = 1:39;
init_var = 0.1;

const_h = 7;
const_w = 7;
channels = 1:176;
const_c = length(channels);
% channels = 1; const_c = 1;

visible = const_h * const_w * const_c;
hidden = visible;
hidden = 1;

const_iteration = 10000; % 10000
const_samples = 300; % 300
const_eval_samples = 1;

SPdata = 1;
method = 2;
if method == 1
    const_gibbs_positive = 30 * hidden;
    const_gibbs_negative = 3 * (visible + hidden);
elseif method == 2
    const_cd_0 = 10; % 10
    const_cd_1 = 7; % 7
    const_cd_2 = 7; % 7
elseif method == 3
    const_variational = 30;
    const_persistentMC = 3;
end

lr_decay = [0.3, 0.6, 0.7, 0.8, 0.9];
print_iteration = 100;
energy_iteration = 10000;
store_iteration = 10000;

% type = 2;
% type = 4;
type = 1;

fix_w12 = 1e0;
bias = 0.0;
disp_scale = 10;