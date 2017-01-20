if ~ exist('features', 'var')
    features = load('../From_zhishuai/res_info.mat');
end
if ~ exist('centers', 'var')
    centers = load('../From_zhishuai/dictionary_imagenet_car_vgg16_pool4_K176_norm_nowarp_prune_512.mat');
end

const_h = 5;
const_w = 5;
% const_c = size(centers, 2);
const_c = 1;

visible = const_h * const_w * const_c;
hidden = 25;

const_iteration = 100000;
const_step = 100;
const_gibbs_positive = 10 * hidden;
const_gibbs_negative = 10 * (visible + hidden);
learning_rate = 0.01;
% 3 * 3 * 176 = 1584

matrix = randn(visible + hidden + 1, visible + hidden + 1);
matrix = 0.5 * matrix + 0.5 * matrix';
mask = ones(visible + hidden + 1, visible + hidden + 1);
for temp = 1:visible + hidden + 1
    matrix(temp, temp) = 0;
    mask(temp, temp) = 0;
end
type = 4;

if type ==0 % Type 0: General BM
elseif type == 1 % Type 1: no visible-visible
    matrix(1:visible,1:visible) = 0;
    mask(1:visible,1:visible) = 0;
elseif type == 2 % Type 2: only visible-visible units
    matrix(1:visible + hidden + 1, visible + 1: visible + hidden) = 0;
    mask(1:visible + hidden + 1, visible + 1: visible + hidden) = 0;
    matrix(visible + 1: visible + hidden, 1:visible + hidden + 1) = 0;
    mask(visible + 1: visible + hidden, 1:visible + hidden + 1) = 0;
elseif type == 3 % Type 3: Restricted BM
    matrix(1:visible,1:visible) = 0;
    mask(1:visible,1:visible) = 0;
    matrix(visible + 1: visible + hidden, visible + 1: visible + hidden) = 0;
    mask(visible + 1: visible + hidden, visible + 1: visible + hidden) = 0;
elseif type == 4 && visible == hidden % Type 4: Wenhao
    matrix(1:visible,1:visible) = 0;
    mask(1:visible,1:visible) = 0;
    matrix(1: hidden, visible + 1: visible + hidden) = eye(hidden);
    mask(1: hidden, visible + 1: visible + hidden) = zeros(hidden);
    matrix(visible + 1: visible + hidden, 1: hidden) = eye(hidden);
    mask(visible + 1: visible + hidden, 1: hidden) = zeros(hidden);
end

h = randn(hidden, 1) > 0.5;
figure();

for iteration = 1:const_iteration
    % disp(['iteration: ', num2str(iteration)]);
    img_idx = randi(length(features.res_info));
    % img_idx = iteration;
    % img_idx = 1;
    % disp(img_idx);
    feature = features.res_info{img_idx}.layer_feature_ori;
    dist = features.res_info{img_idx}.layer_feature_dist;
    [height, width, ~] = size(dist);
    % imshow(dist(1:h, 1:w, 1));
    
    top = randi(height - const_h + 1);
    left = randi(width - const_w + 1);
    crop = dist( top: top + const_h - 1, left: left + const_w - 1, 1:const_c );
    
    % input_tensor = 2 ./ (1 + exp(crop)); % non linear
    % input_tensor = 1 - 0.5 .* crop; % linear
    input_tensor = crop < 1; % magic threshold
    % distance to probability function to explore (generate spikes)
    
    v = reshape(input_tensor, visible, 1);
    
    v_indices = 1:visible;
    h_indices = visible + 1 : visible + hidden;
    vh_indices = [v_indices, h_indices];
    vhb = [v; h; 1];
    
    % positive step
    for gibbs= 1:const_gibbs_positive
        index = h_indices(randi(numel(h_indices)));
        result = matrix(index, :) * vhb;
        prob = 1 ./ (1 + exp(-result));
        vhb(index) = rand() < prob;
    end
    positive_vhb = vhb;
    
    % negative step
    for gibbs= 1:const_gibbs_negative
        index = vh_indices(randi(numel(vh_indices)));
        result = matrix(index, :) * [v; h; 1];
        prob = 1 ./ (1 + exp(-result));
        vhb(index) = rand() < prob;
    end
    negative_vhb = vhb;
    
    positive_dmatrix = positive_vhb * positive_vhb';
    negative_dmatrix = negative_vhb * negative_vhb';
    dmatrix = positive_dmatrix - negative_dmatrix;
    
    matrix = matrix + learning_rate * dmatrix .* mask;
    disp(['iteration ', num2str(iteration), ': ', num2str(norm(dmatrix))]); 
    
%     last_mvn = v;
%     for step= 1:const_step
%         mvn = 1 ./ (1 + exp(-last_matrix_V * [last_mvn; 1]));
%         % disp(['step ', num2str(step), ':', num2str(norm(mvn - last_mvn))]);
%         last_mvn = mvn;
%     end
%     dV = v * [v;1]' - mvn * [mvn;1]';
%     matrix_V = last_matrix_V + learning_rate * dV / iteration;
%     disp(['iteration ', num2str(iteration), ': ', num2str(norm(dV))]);
%     last_matrix_V = matrix_V;
    
%     distance = zeros(h,w,const_c);
%     for hh = 1:h
%         for ww = 1:w
%             slice = reshape(feature(hh,ww,:), [512, 1]);
%             slice = slice / norm(slice, 2);
%             temp = centers - repmat(slice, [1, const_c]);
%             temp = sum(temp .^ 2, 1)
%         end
%     end
end
imshow(kron(matrix,ones(10)), [-3, 3], 'Colormap', jet(256));
