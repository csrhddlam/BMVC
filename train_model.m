
v_indices = 1:visible;
h_indices = visible + 1 : visible + hidden;
vh_indices = [v_indices, h_indices];

% training_indices = find((partition == 1) .* ((label == label_train(1)) + (label == label_train(2))));
% validation_indices = find((partition == 2) .* ((label == label_train(1)) + (label == label_train(2))));
% cheat_indices = find((label ~= label_train(1)) .* (label ~= label_train(2)));

training_indices = find((partition == 1) .* (label == label_train));
validation_indices = find((partition == 2) .* (label == label_train));
cheat_indices = find(label ~= label_train);

cheat_indices = datasample(cheat_indices, 1000);

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

training_data = gpuArray(single(data(training_indices, :)));
validation_data = gpuArray(single(data(validation_indices, :)));
cheat_data = gpuArray(single(data(cheat_indices, :)));
% training_data = gpuArray(single(zeros(length(training_indices), visible)));
% validation_data = gpuArray(single(zeros(length(validation_indices), visible)));
% cheat_data = gpuArray(single(zeros(length(cheat_indices), visible)));
%% prepare training data
% for index = 1:length(training_indices)
%     train_index = training_indices(index);
%     training_data(index, :) = get_data_from_index(data_all, train_index, const_h, const_w, channels)';
% end
% for index = 1:length(validation_indices)
%     val_index = validation_indices(index);
%     validation_data(index, :) = get_data_from_index(data_all, val_index, const_h, const_w, channels)';
% end
% for index = 1:length(cheat_indices)
%     cheat_index = cheat_indices(index);
%     cheat_data(index, :) = get_data_from_index(data_all, cheat_index, const_h, const_w, channels)';
% end

for iteration = 1:const_iteration
%% prepare mini-batch data
    vn = datasample(training_data, const_samples, 'Replace', true)';

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

    %% negative steps

        for gibbs = 1:const_gibbs_negative
            index = vh_indices(randi(numel(vh_indices)));
            result = matrix(index, :) * negative_vhbs;
            prob = 1 ./ (1 + exp(-result));
            negative_vhbs(index,:) = rand() < prob;
        end
        dmatrix_negative = negative_vhbs * negative_vhbs' / const_samples;
        dmatrix = dmatrix_positive - dmatrix_negative;
    elseif method == 2
    %% contrastive divergence mean field
        mhn = gpuArray(single(ones(hidden, const_samples) * 0.5));
        for i = 1:const_cd_0% * hidden
%             index = randi(hidden);
            index = 1:hidden;
            temp = matrix(visible + index, :) * [vn; mhn; ones(1, const_samples)];
            mhn(index, :) = (1 - momentum) * mhn(index, :) + momentum * 1 ./ (1 + exp(-temp));
        end
        mvn = vn;
        for i = 1:const_cd_1% * visible
%             index = randi(visible);
            index = 1:visible;
            temp = matrix(index, :) * [mvn; mhn; ones(1, const_samples)];
            mvn(index, :) = (1 - momentum) * mvn(index, :) + momentum * 1 ./ (1 + exp(-temp));
        end
        mhnt = mhn;
        for i = 1:const_cd_2% * hidden
%             index = randi(hidden);
            index = 1:hidden;
            temp = matrix(visible + index, :) * [mvn; mhnt; ones(1, const_samples)];
            mhnt(index, :) = (1 - momentum) * mhnt(index, :) + momentum * 1 ./ (1 + exp(-temp));
        end
        
        dW = (mhn * mhn' - mhnt * mhnt') ./ const_samples - weight_decay .* matrix(visible+1:visible+hidden, visible+1:visible+hidden);
        dV = (vn * vn' - mvn * mvn') ./ const_samples - weight_decay .* matrix(1:visible, 1:visible);
        dJ = (vn * mhnt' - mvn * mhnt') ./ const_samples - weight_decay .* matrix(1:visible, visible+1:visible+hidden);
        dBv = mean(vn - mvn, 2); % no wd for bias
        dBh = mean(mhn - mhnt, 2); % no wd for bias
        
        dmatrix = [dV, dJ, dBv; dJ', dW, dBh; dBv', dBh', 0];
    elseif method == 3
        
    end
%% updates
    temp_lr = 0.1 ^ sum((iteration / const_iteration) > lr_decay) * learning_rate;
    dmatrix = dmatrix .* mask;
    matrix = matrix + dmatrix .* temp_lr;
%% print
    if mod(iteration, print_iteration) == 0
%         visualize_model(matrix, visible, hidden, const_h, fig1, fig2, fig3, disp_scale);
        matrix_sum = gather(sum(sum(abs(dmatrix))));
        matrix_max = gather(max(max(abs(dmatrix))));
        disp([datestr(now), ' iteration ', num2str(iteration), ', learning_rate ', num2str(temp_lr), ', gradient_sum ', num2str(matrix_sum), ', gradient_max ', num2str(matrix_max)]);
        
        gradient_history(history_i) = log10(max(matrix_sum, 0.0000001));
%         training_free_energy_history(history_i) = training_free_energy;
%         validation_free_energy_history(history_i) = validation_free_energy;
%         cheat_free_energy_history(history_i) = cheat_free_energy;
        history_i = history_i + 1;
        % figure(fig_history);
%         plot([training_free_energy_history;validation_free_energy_history;cheat_free_energy_history]');
%         legend('training', 'validation', 'cheat'); drawnow;
%         plot(gradient_history); drawnow;
    end
    %% all visible case free energy over all training data
    if mod(iteration, energy_iteration) == 0 && (type == 1 || type == 2)
        training_free_energies = 99999999 * ones(length(training_indices), 1);
        for index = 1:length(training_indices)
            visible_binary = training_data(index, :);
            for hidden_int = 0:2^hidden-1
                hidden_binary = de2bi(hidden_int, hidden, 'left-msb');
                temp_data = [visible_binary, hidden_binary, 1];
                training_free_energies(index) = min(training_free_energies(index), gather(-temp_data * matrix * temp_data') / 2);
            end
        end
        training_free_energy = mean(training_free_energies / 2^hidden);

        validation_free_energies = 99999999 * ones(length(validation_indices), 1);
        for index = 1:length(validation_indices)
            visible_binary = validation_data(index, :);
            for hidden_int = 0:2^hidden-1
                hidden_binary = de2bi(hidden_int, hidden, 'left-msb');
                temp_data = [visible_binary, hidden_binary, 1];
                validation_free_energies(index) = min(validation_free_energies(index), gather(-temp_data * matrix * temp_data') / 2);
            end
        end
        validation_free_energy = mean(validation_free_energies / 2^hidden);

        cheat_free_energies = 99999999 * ones(length(cheat_indices), 1);
        for index = 1:length(cheat_indices)
            visible_binary = cheat_data(index, :);
            for hidden_int = 0:2^hidden-1
                hidden_binary = de2bi(hidden_int, hidden, 'left-msb');
                temp_data = [visible_binary, hidden_binary, 1];
                cheat_free_energies(index) = min(cheat_free_energies(index), gather(-temp_data * matrix * temp_data') / 2);
            end
        end
        cheat_free_energy = mean(cheat_free_energies / 2^hidden);
        disp(['training ', num2str(training_free_energy), ', validation ', num2str(validation_free_energy), ', cheat ', num2str(cheat_free_energy)]);
    end
    %% save model
    if mod(iteration, store_iteration) == 0
        save(['matrix_', num2str(label_train), '_lr_', num2str(learning_rate),'_mom_', num2str(momentum), '_hidden_',num2str(hidden),'.mat'],'matrix');
    end
%% break point
    if iteration == 1000000
        a = 0;
    end
end

% colorbar;
% imshow(kron(matrix,ones(10)), [-50, 50], 'Colormap', jet(256));