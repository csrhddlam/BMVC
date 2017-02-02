function score = interface_zhishuai(input_7_7_176, matrix, visible, hidden)
    binary_tensor = single(input_7_7_176 < 0.65);
    visible_units = reshape(binary_tensor, numel(binary_tensor), 1);
    hidden_units = double(randn(hidden, 1) < 0.5);
    probability = [visible_units; hidden_units]' * matrix(1:visible+hidden,:) * [visible_units; hidden_units;1];
    score = gather(probability);
end