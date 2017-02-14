data_train_SVM_pos = find((partition_all == 1) .* (label_all == label_train));
data_train_SVM_neg = find((partition_all == 1) .* (label_all ~= label_train));

data_test_SVM_pos = find((partition_all == 2) .* (label_all == label_train));
data_test_SVM_neg = find((partition_all == 2) .* (label_all ~= label_train));

max_train_pos = 261;
max_train_neg = 261;

max_test_pos = 81;
max_test_neg = 81;

real_train_pos = min(length(data_train_SVM_pos), max_train_pos);
real_train_neg = min(length(data_train_SVM_neg), max_train_neg);
real_test_pos = min(length(data_test_SVM_pos), max_test_pos);
real_test_neg = min(length(data_test_SVM_neg), max_test_neg);

feature_length = numel(data_all{1});
train_SVM_pos = zeros(real_train_pos, feature_length);
train_SVM_neg = zeros(real_train_neg, feature_length);
test_SVM_pos = zeros(real_test_pos, feature_length);
test_SVM_neg = zeros(real_test_neg, feature_length);

index_train_pos = 1;
index_train_neg = 1;
index_test_pos = 1;
index_test_neg = 1;

for i = randperm(length(data_train_SVM_pos), real_train_pos)
    train_SVM_pos(index_train_pos, :) = reshape(double(data_all{data_train_SVM_pos(i)} < 0.65), 1, feature_length);
    index_train_pos = index_train_pos + 1;
end
for i = randperm(length(data_train_SVM_neg), real_train_neg)
    train_SVM_neg(index_train_neg, :) = reshape(double(data_all{data_train_SVM_neg(i)} < 0.65), 1, feature_length);
    index_train_neg = index_train_neg + 1;
end
for i = randperm(length(data_test_SVM_pos), real_test_pos)
    test_SVM_pos(index_test_pos, :) = reshape(double(data_all{data_test_SVM_pos(i)} < 0.65), 1, feature_length);
    index_test_pos = index_test_pos + 1;
end
for i = randperm(length(data_test_SVM_neg), real_test_neg)
    test_SVM_neg(index_test_neg, :) = reshape(double(data_all{data_test_SVM_neg(i)} < 0.65), 1, feature_length);
    index_test_neg = index_test_neg + 1;
end

train_SVM_data = [train_SVM_pos; train_SVM_neg];
train_SVM_label = [ones(real_train_pos, 1); zeros(real_train_neg, 1)];
test_SVM_data = [test_SVM_pos; test_SVM_neg];
test_SVM_label = [ones(real_test_pos, 1); zeros(real_test_neg, 1)];

% train_SVM_data = feature_transform(train_SVM_data);
% test_SVM_data = feature_transform(test_SVM_data);

% trainSVM_data = [1,1,2,1;2,2,3,2;0,0,1,0;-1,-1,-1,-2];
% trainSVM_label = [1;1;0;0];
model = svmtrain(train_SVM_label, train_SVM_data, '-t 0');
[train_predicted_label, train_accuracy, train_decision_values] = svmpredict(train_SVM_label, train_SVM_data, model);
[test_predicted_label, test_accuracy, test_decision_values] = svmpredict(test_SVM_label, test_SVM_data, model);
weights = model.sv_coef' * model.SVs;
bias = -model.rho;
my_decision_values = test_SVM_data * weights' + bias;