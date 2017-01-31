train_data_meanP = zeros(2000, 1);
val_data_meanP = zeros(1231, 1);
test_data_meanP = zeros(3018, 1);


probes = cell(3, 1);

probes{1} = find((partition_all == 1) .* (label_all == label_train));
probes{2} = find((partition_all == 2) .* (label_all == label_train));
probes{3} = find((partition_all == 2) .* (label_all == label_val));


results_raw = cell(size(probes));
results_mean = cell(size(probes));

results_min = 99999999;
results_max = -99999999;
for probe_index = 1:length(probes) % for each probe
    disp(probe_index);
    eval_indices = probes{probe_index};
    results_raw{probe_index} = zeros(length(eval_indices), const_eval_samples);
    for eval_index_index = 1:length(eval_indices) % for each data point
        eval_index = eval_indices(eval_index_index);
        for sample_index = 1:const_eval_samples
            visible_units = get_data_from_index(data_all, eval_index, const_h, const_w, channels);
            hidden_units = double(randn(hidden, 1) < 0.5);
            probability = [visible_units; hidden_units]' * matrix(1:visible+hidden,:) * [visible_units; hidden_units;1];
            results_raw{probe_index}(eval_index_index, sample_index) = gather(probability);
        end
    end
    results_mean{probe_index} = mean(results_raw{probe_index}, 2);
    results_min = min(results_min, min(results_mean{probe_index}));
    results_max = max(results_max, max(results_mean{probe_index}));
end

hist_bins = results_min:10:results_max;
results_hist = zeros(length(probes),length(hist_bins));
for probe_index = 1:length(probes)
    results_hist(probe_index,:) = hist(results_mean{probe_index}, hist_bins);
end
plot(hist_bins, results_hist);

