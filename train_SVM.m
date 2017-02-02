data_trainSVMpos = find((partition_all == 1) .* (label_all == label_train));
data_trainSVMneg = find((partition_all == 1) .* (label_all ~= label_train));

max_pos = 1000;
max_neg = 1000;

real_pos = min(length(data_trainSVMpos), max_pos);
real_neg = min(length(data_trainSVMneg), max_neg);

feature_length = numel(data_all{1});
trainSVMpos = zeros(real_pos, feature_length);
trainSVMneg = zeros(real_neg, feature_length);

index_pos = 1;
for i = randperm(length(data_trainSVMpos), real_pos)
    trainSVMpos(index_pos, :) = reshape(double(data_all{data_trainSVMpos(i)} < 0.65), 1, feature_length);
    index_pos = index_pos + 1;
end
index_neg = 1;
for i = randperm(length(data_trainSVMneg), real_neg)
    trainSVMneg(index_neg, :) = reshape(double(data_all{data_trainSVMneg(i)} < 0.65), 1, feature_length);
    index_neg = index_neg + 1;
end

model = svmtrain([trainSVMpos; trainSVMneg], [ones(real_pos, 1); -ones(real_neg, 1)], '');
[predicted_label, accuracy, decision_values] = svmpredict([ones(real_pos, 1); -ones(real_neg, 1)], [trainSVMpos; trainSVMneg], model, '');
