
v_indices = 1:visible;
h_indices = visible + 1 : visible + hidden;
vh_indices = [v_indices, h_indices];
training_indices = find((partition_all == 1) .* (label_all == label_train));
validation_indices = find((partition_all == 2) .* (label_all == label_train));
cheat_indices = find(label_all ~= label_train);
cheat_indices = datasample(cheat_indices, const_samples);

positive_vhbs = double(randn(visible + hidden + 1, const_samples) < 0.1);
positive_vhbs(visible + hidden + 1, :) = 1;

negative_vhbs = double(randn(visible + hidden + 1, const_samples) < 0.1);
negative_vhbs(visible + hidden + 1, :) = 1;

persistent_samples = randn(visible + hidden, const_samples);

gradient_history = zeros(1, const_iteration / print_iteration);
training_free_energy_history = zeros(1, const_iteration / print_iteration);
validation_free_energy_history = zeros(1, const_iteration / print_iteration);
cheat_free_energy_history = zeros(1, const_iteration / print_iteration);
history_i = 1;

training_data = gpuArray(single(zeros(length(training_indices), visible)));
validation_data = gpuArray(single(zeros(length(validation_indices), visible)));
cheat_data = gpuArray(single(zeros(length(cheat_indices), visible)));
%% prepare training data
for index = 1:length(training_indices)
    train_index = training_indices(index);
    training_data(index, :) = get_data_from_index(data_all, train_index, const_h, const_w, channels)';
end
for index = 1:length(validation_indices)
    val_index = validation_indices(index);
    validation_data(index, :) = get_data_from_index(data_all, val_index, const_h, const_w, channels)';
end
for index = 1:length(cheat_indices)
    cheat_index = cheat_indices(index);
    cheat_data(index, :) = get_data_from_index(data_all, cheat_index, const_h, const_w, channels)';
end
for iteration = 1:const_iteration
%% prepare mini-batch data
    vn = datasample(training_data, const_samples, 'Replace', false)';

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
    elseif method == 2
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
        dBv = mean(vn - mvn, 2) ;
        dBh = mean(mhn - mhnt, 2) ;
        
        dmatrix = [dV, dJ, dBv; dJ', dW, dBh; dBv', dBh', 0];
    elseif method == 3
        
    end
%% updates
    temp_lr = 0.1 ^ floor(iteration / half_life_iteration) * learning_rate;
    dmatrix = dmatrix .* mask;
    matrix = matrix + temp_lr * dmatrix;
    matrix = matrix * weight_decay;
    if mod(iteration, print_iteration) == 0
%         visualize_model(matrix, visible, hidden, const_h, fig1, fig2, fig3, disp_scale);
        %% all visible case free energy over all training data
        if type == 2
            training_free_energies = zeros(length(training_indices), 1);
            for index = 1:length(training_indices)
                training_free_energies(index) = gather(-training_data(index, :) * matrix(1:visible, :) * [training_data(index, :), 0, 1]');
            end
            training_free_energy = mean(training_free_energies)
            
            validation_free_energies = zeros(length(validation_indices), 1);
            for index = 1:length(validation_indices)
                validation_free_energies(index) = gather(-validation_data(index, :) * matrix(1:visible, :) * [validation_data(index, :), 0, 1]');
            end
            validation_free_energy = mean(validation_free_energies)
            
            cheat_free_energies = zeros(length(cheat_indices), 1);
            for index = 1:length(cheat_indices)
                cheat_free_energies(index) = gather(-cheat_data(index, :) * matrix(1:visible, :) * [cheat_data(index, :), 0, 1]');
            end
            cheat_free_energy = mean(cheat_free_energies)
        end
        
        matrix_sum = gather(sum(sum(abs(dmatrix))));
        matrix_max = gather(max(max(abs(dmatrix))));
        disp([datestr(now), ' iteration ', num2str(iteration), ', learning_rate ', num2str(temp_lr), ', gradient_sum ', num2str(matrix_sum), ', gradient_max ', num2str(matrix_max)]);
        disp(['training ', num2str(training_free_energy), ', validation ', num2str(validation_free_energy), ', cheat ', num2str(cheat_free_energy)]);

        save(['matrix_', num2str(label_train),'_lr_', num2str(learning_rate),'_mom_', num2str(momentum),'.mat'],'matrix');
        
        gradient_history(history_i) = log10(max(matrix_sum, 0.0000001));
        training_free_energy_history(history_i) = training_free_energy;
        validation_free_energy_history(history_i) = validation_free_energy;
        cheat_free_energy_history(history_i) = cheat_free_energy;
        history_i = history_i + 1;
        % figure(fig_history);
        plot([training_free_energy_history;validation_free_energy_history;cheat_free_energy_history]');
        legend('training', 'validation', 'cheat'); drawnow;
%         plot(gradient_history); drawnow;
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