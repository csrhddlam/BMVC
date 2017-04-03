function [data, partition, label] = read_data_python()
    load('data.mat');
    data = data';
    label = label';
    total_length = length(label);
    partition = floor(randi(5, total_length, 1) / 5) + 1;
end