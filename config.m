%% constants
label_train = 1;
label_val = 1:39;

const_h = 7;
const_w = 7;
% channel = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17];
% channel = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
channels = 1:176;
const_c = length(channels);
% channel = 1; const_c = 1;

visible = const_h * const_w * const_c;
hidden = visible;
hidden = 1;

const_iteration = 100000;
const_samples = 100;
const_eval_samples = 1;

SPdata = 1;
method = 2;
if method == 1
    const_gibbs_positive = 30 * hidden;
    const_gibbs_negative = 3 * (visible + hidden);
elseif method == 2
    const_cd_0 = 30;
    const_cd_1 = 1;
    const_cd_2 = 5;
elseif method == 3
    const_variational = 30;
    const_persistentMC = 3;
end

learning_rate = 1e0;
momentum = 0.1;
half_life_iteration = 1e-1 * const_iteration;
print_iteration = 100;

% type = 2;
% type = 4;
type = 2;

w12 = 1e0;
bias = 0.0;
disp_scale = 10;