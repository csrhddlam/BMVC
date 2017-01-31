SP_indices = [label_train, label_val];
train_sample = 0.6;
val_sample = 0.2;
test_sample = 0.2;

length_all = 0;

% if ~ exist('features', 'var')
%     features = load('../From_zhishuai/res_info.mat');
% end
% if ~ exist('centers', 'var')
%     centers = load('../From_zhishuai/dictionary_imagenet_car_vgg16_pool4_K176_norm_nowarp_prune_512.mat');
% end

if ~ exist('data_all', 'var')
    file_list = dir('../From_zhishuai/spFeat/*.mat');
    data_raw = cell(length(file_list), 1);
    for f = 1:length(file_list)
        data_raw{f} = load(['../From_zhishuai/spFeat/', file_list(f).name]);
        for SP_index = SP_indices
            length_all = length_all + length(data_raw{f}.featSP{SP_index});
        end
    end
    index = 1;
    data_all = cell(length_all, 1);
    label_all = zeros(length_all, 1);
    partition_all = zeros(length_all, 1);
    
    for f = 1:length(data_raw)
        for SP_index = SP_indices
            for instance = 1:length(data_raw{f}.featSP{SP_index})
                data_all{index} = data_raw{f}.featSP{SP_index}{instance};
                label_all(index) = SP_index;
                luck = rand();
                partition_all(index) = (luck > train_sample) + (luck > train_sample + val_sample) + 1;
                index = index + 1;
            end
        end
    end
    clear file_list;
    clear data_raw;
end