function output = get_data_from_index(data_all, index, const_h, const_w, channels)
    dist = data_all{index};
    [height, width, ~] = size(dist);
    % top = randi(height - const_h + 1);
    % left = randi(width - const_w + 1);
    top = round((height - const_h + 1) / 2);
    left = round((width - const_w + 1) / 2);
    crop = dist( top: top + const_h - 1, left: left + const_w - 1, channels );
    input_tensor = double(crop < 0.65);
    output = reshape(input_tensor, numel(input_tensor), 1);
end