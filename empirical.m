label_counting = 1;
indices = find((partition_all == 1) .* (label_all == label_counting));

counting = ones(visible, 1);
for eval_index_index = 1:length(indices) % for each data point
    eval_index = indices(eval_index_index);
    visible_units = get_data_from_index(data_all, eval_index, const_h, const_w, channels);
    counting = counting + visible_units;
end

% w = gather(matrix(visible+hidden+1, 1:visible));
counting = counting / length(indices);
w_counting = log(counting ./ (1 - counting))';
matrix(visible+hidden+1, 1:visible) = w_counting;
matrix(1:visible, 1+visible+hidden) = w_counting;
matrix = max(matrix, -10);
matrix = gpuArray(single(matrix));
save(['matrix_', num2str(label_counting),'_counting.mat'],'matrix');
a = 0;

figure;
imshow(reshape(weights, 7, 8624 / 7), []);
figure;
imshow(reshape(counting, 7, 8624 / 7), []);
