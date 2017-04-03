
% read_data;
[data, partition, label] = read_data_python();

% label_trains = zeros(4, 2);
% label_trains(1,:) = [2,3];
% label_trains(2,:) = [22, 23];
% label_trains(3,:) = [38, 39];
% label_trains(4,:) = [1, 23];
label_trains = 1:40;
label_trains = label_trains';

for i = 1:size(label_trains, 1)
%     label_train = label_trains(i, :);
    label_train = label_trains(i);
    disp(label_train);
    config;
    init;
    train_model;
    % eval_model;
end