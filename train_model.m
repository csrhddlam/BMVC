
v_indices = 1:visible;
h_indices = visible + 1 : visible + hidden;
vh_indices = [v_indices, h_indices];

positive_vhbs = double(randn(visible + hidden + 1, const_samples) < 0.1);
positive_vhbs(visible + hidden + 1, :) = 1;

negative_vhbs = double(randn(visible + hidden + 1, const_samples) < 0.1);
negative_vhbs(visible + hidden + 1, :) = 1;

vn = gpuArray(single(randn(visible, const_samples) < 0.1));
% mvn = double(randn(visible, const_samples) < 0.1);
% mhnt = double(randn(hidden, const_samples) < 0.1);

train_indices = find((partition_all == 1) .* (label_all == label_train));
for iteration = 1:const_iteration
%% prepare mini-batch data
    for index = 1:const_samples
        if SPdata == 1
            train_index = train_indices(randi(length(train_indices)));
            vn(1:visible, index) = get_data_from_index(data_all, train_index, const_h, const_w, channels);
        else
            img_idx = randi(length(features.res_info));
            % img_idx = 1;
            % disp(img_idx);
            % feature = features.res_info{img_idx}.layer_feature_ori;
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
    end

    if method == 1
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
        
        mhn = gpuArray(single(ones(hidden, const_samples) * 0.5));
        for i = 1:const_cd_0% * hidden
%             index = randi(hidden);
            index = 1:hidden;
%             Energy_positive = W(index, :) * mhn + JT(index, :) * vn + repmat(Bh(index, :), 1, const_samples);
            temp = matrix(visible + index, :) * [vn; mhn; ones(1, const_samples)];
            mhn(index, :) = (1 - momentum) * mhn(index, :) + momentum * 1 ./ (1 + exp(-temp));
        end
        mvn = vn;
        for i = 1:const_cd_1% * visible
%             temp = V * mvn + J * mhn + repmat(Bv, 1, const_samples);
%             index = randi(visible);
            index = 1:visible;
            temp = matrix(index, :) * [mvn; mhn; ones(1, const_samples)];
            mvn(index, :) = (1 - momentum) * mvn(index, :) + momentum * 1 ./ (1 + exp(-temp));
        end
        mhnt = mhn;
        for i = 1:const_cd_2% * hidden
%             index = randi(hidden);
            index = 1:hidden;
%             temp = W(index, :) * mhnt + JT(index, :) * mvn + repmat(Bh(index, :), 1, const_samples);
            temp = matrix(visible + index, :) * [mvn; mhnt; ones(1, const_samples)];
            mhnt(index, :) = (1 - momentum) * mhnt(index, :) + momentum * 1 ./ (1 + exp(-temp));
        end
        
        dW = (mhn * mhn' - mhnt * mhnt') / const_samples;
        dV = (vn * vn' - mvn * mvn') / const_samples;
        dJ = (vn * mhnt' - mvn * mhnt') / const_samples;
        dBv = mean(vn - mvn, 2);
        dBh = mean(mhn - mhnt, 2);
        
        dmatrix = [dV, dJ, dBv; dJ', dW, dBh; dBv', dBh', 0];
    end
%% updates
    temp_lr = learning_rate * half_life_iteration / (half_life_iteration + iteration);
    matrix = matrix + temp_lr * dmatrix .* mask;
    if mod(iteration, print_iteration) == 0
        disp(['iteration ', num2str(iteration), ': ', num2str(norm(dmatrix))]);
        disp(datestr(now));
        save('matrix.mat','matrix');
%         disp(['    delta_free_energy :', num2str(delta_free_energy(iteration))]);
%         disp(['        free_energy_positive :', num2str(free_energy_positive)]);
%         disp(['        free_energy_negative :', num2str(free_energy_negative)]);
        %visualize_model(matrix, visible, hidden, const_h, fig1, fig2, fig3, disp_scale);
        %figure(fig3);
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
%         ends
%     end
%% break point
    if iteration == 1000000
        a = 0;
    end
end

colorbar;
% imshow(kron(matrix,ones(10)), [-50, 50], 'Colormap', jet(256));