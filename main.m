%% load
if ~ exist('features', 'var')
    features = load('../From_zhishuai/res_info.mat');
end
if ~ exist('centers', 'var')
    centers = load('../From_zhishuai/dictionary_imagenet_car_vgg16_pool4_K176_norm_nowarp_prune_512.mat');
end

%% constants
const_h = 5;
const_w = 5;
% const_c = size(centers, 2);
channel = [1, 9, 16, 22, 25];
const_c = length(channel);

visible = const_h * const_w * const_c;
hidden = visible;
% hidden = 1;

const_iteration = 100000;
const_samples = 100;

const_gibbs_positive = 30 * hidden;
const_gibbs_negative = 3 * (visible + hidden);

const_cd_0 = 30;
const_cd_1 = 3;
const_cd_2 = 5;

learning_rate = 1e0;
half_life_iteration = 1e-1 * const_iteration;
print_iteration = 10;

type = 4;
MFCD = 1;
w12 = 1e0;
bias = 0.0;
disp_scale = 10;

delta_free_energy = zeros(1, const_iteration);
figure;

%% matrix init

% matrix = randn(visible + hidden + 1, visible + hidden + 1) * 1;
% matrix = 0.5 * matrix + 0.5 * matrix'; % symmetry

mask = ones(visible + hidden + 1, visible + hidden + 1);
for temp = 1:visible + hidden + 1
    matrix(temp, temp) = 0;
    mask(temp, temp) = 0;
end

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
elseif type == 4 && hidden == visible % Type 4: Wenhao
    matrix(1:visible,1:visible) = 0;
    mask(1:visible,1:visible) = 0;
    matrix(1: hidden, visible + 1: visible + hidden) = eye(hidden) * w12;
    mask(1: hidden, visible + 1: visible + hidden) = zeros(hidden);
    matrix(visible + 1: visible + hidden, 1: hidden) = eye(hidden) * w12;
    mask(visible + 1: visible + hidden, 1: hidden) = zeros(hidden);
    % matrix(visible + 1: visible + hidden, visible + 1: visible + hidden) = matrix_init;
end


%% for loop
v_indices = 1:visible;
h_indices = visible + 1 : visible + hidden;
vh_indices = [v_indices, h_indices];

positive_vhbs = double(randn(visible + hidden + 1, const_samples) < 0.1);
positive_vhbs(visible + hidden + 1, :) = 1;

negative_vhbs = double(randn(visible + hidden + 1, const_samples) < 0.1);
negative_vhbs(visible + hidden + 1, :) = 1;

vn = double(randn(visible, const_samples) < 0.1);
mvn = double(randn(visible, const_samples) < 0.1);
mhnt = double(randn(hidden, const_samples) < 0.1);

for iteration = 1:const_iteration
%% prepare mini-batch data
    for index = 1:const_samples
        % img_idx = randi(length(features.res_info));
        img_idx = randi(1000);
        % img_idx = 1;
        % disp(img_idx);
        feature = features.res_info{img_idx}.layer_feature_ori;
        dist = features.res_info{img_idx}.layer_feature_dist;
        [height, width, ~] = size(dist);
        % imshow(dist(1:h, 1:w, 1));

        top = randi(height - const_h + 1);
        left = randi(width - const_w + 1);
%         top = height - const_h + 1;
%         left = round((width - const_w + 2) / 2);
        crop = dist( top: top + const_h - 1, left: left + const_w - 1, channel );

        % input_tensor = 2 ./ (1 + exp(crop)); % non linear
        % input_tensor = 1 - 0.5 .* crop; % linear
        input_tensor = double(crop < 1); % threshold to binary
        % distance to probability function to explore (generate spikes)
        vn(1:visible, index) = reshape(input_tensor, visible, 1);
    end

    if ~MFCD
    %% positive gibbs
        positive_vhbs(1:visible, 1:const_samples) = vn;
        for gibbs = 1:const_gibbs_positive
            index = h_indices(randi(numel(h_indices)));
            result = matrix(index, :) * positive_vhbs;
            prob = 1 ./ (1 + exp(-result));
            positive_vhbs(index,:) = rand() < prob;
        end
        dmatrix_positive = positive_vhbs * positive_vhbs' / const_samples;
%         temp_vhbs = positive_vhbs;
%         temp_vhbs(visible+1:visible+hidden, 1:const_samples) = double(randn(hidden, const_samples) < 0.5);
%         energy_positive = -diag(temp_vhbs' * matrix * temp_vhbs);
%         free_energy_positive = -log(mean(exp(-energy_positive)));

    %% negative steps

        for gibbs = 1:const_gibbs_negative
            index = vh_indices(randi(numel(vh_indices)));
            result = matrix(index, :) * negative_vhbs;
            prob = 1 ./ (1 + exp(-result));
            negative_vhbs(index,:) = rand() < prob;
        end
        dmatrix_negative = negative_vhbs * negative_vhbs' / const_samples;
%         temp_vhbs = positive_vhbs;
%         temp_vhbs(visible+1:visible+hidden, 1:const_samples) = double(randn(hidden, const_samples) < 0.5);
%         energy_negative = -diag(temp_vhbs' * matrix * temp_vhbs);
%         free_energy_negative = -log(mean(exp(-energy_negative)));
        dmatrix = dmatrix_positive - dmatrix_negative;
        % delta_free_energy(iteration) = free_energy_positive - free_energy_negative;
    else
    %% contrastive divergence mean field
        
        mhn = double(ones(hidden, const_samples) * 0.5);
        for i = 1:const_cd_0 * hidden
            index = randi(hidden);
%             Energy_positive = W(index, :) * mhn + JT(index, :) * vn + repmat(Bh(index, :), 1, const_samples);
            temp = matrix(visible + index, :) * [vn; mhn; ones(1, const_samples)];
            mhn(index, :) = 1 ./ (1 + exp(-temp));
        end
        mvn = vn;
        for i = 1:const_cd_1 * visible
%             temp = V * mvn + J * mhn + repmat(Bv, 1, const_samples);
            index = randi(visible);
            temp = matrix(index, :) * [mvn; mhn; ones(1, const_samples)];
            mvn(index, :) = 1 ./ (1 + exp(-temp));
        end
        mhnt = mhn;
        for i = 1:const_cd_2 * hidden
            index = randi(hidden);
%             temp = W(index, :) * mhnt + JT(index, :) * mvn + repmat(Bh(index, :), 1, const_samples);
            temp = matrix(visible + index, :) * [mvn; mhnt; ones(1, const_samples)];
            mhnt(index, :) = 1 ./ (1 + exp(-temp));
        end
        
        dW = (mhn * mhn' - mhnt * mhnt') / const_samples;
        dV = (vn * vn' - mvn * mvn') / const_samples;
        dJ = (vn * mhnt' - mvn * mhnt') / const_samples;
        dBv = mean(vn - mvn, 2);
        dBh = mean(mhn - mhnt, 2);
        
        dmatrix = [dV, dJ, dBv; dJ', dW, dBh; dBv', dBh', 0];
    end
%% updates

    matrix = matrix + learning_rate * half_life_iteration / (half_life_iteration + iteration) * dmatrix .* mask;
    if mod(iteration, print_iteration) == 0
        disp(['iteration ', num2str(iteration), ': ', num2str(norm(dmatrix))]);
%         disp(['    delta_free_energy :', num2str(delta_free_energy(iteration))]);
%         disp(['        free_energy_positive :', num2str(free_energy_positive)]);
%         disp(['        free_energy_negative :', num2str(free_energy_negative)]);
%         figure(1);
        imshow(kron(matrix,ones(floor(888 / (visible + hidden + 1)))), [-disp_scale, disp_scale], 'Colormap', jet(256));
%         figure(2);
%         plot(delta_free_energy);
    end
%% compute distance
%     distance = zeros(h,w,const_c);
%     for hh = 1:h
%         for ww = 1:w
%             slice = reshape(feature(hh,ww,:), [512, 1]);
%             slice = slice / norm(slice, 2);
%             temp = centers - repmat(slice, [1, const_c]);
%             temp = sum(temp .^ 2, 1);
%         end
%     end
%% break point
    if iteration == 1000
        a = 0;
    end
end
colorbar;
% imshow(kron(matrix,ones(10)), [-50, 50], 'Colormap', jet(256));