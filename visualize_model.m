function visualize_model(matrix, visible, hidden, const_h, fig1, fig2, fig3, disp_scale)
    disp(datestr(now));
    cpu_matrix = gather(matrix);
    %% insert row
    const_insert = const_h * const_h;

    output_matrix = cpu_matrix;

    temp_size = [ceil(size(output_matrix,1)/const_insert)*const_insert, size(output_matrix,2)] ;
    temp_matrix = -100 * ones(temp_size);
    temp_matrix(1:size(output_matrix,1),1:size(output_matrix,2)) = output_matrix;

    temp_matrix = reshape(temp_matrix, const_insert, numel(temp_matrix)/const_insert);
    temp_matrix = [temp_matrix; -100 * ones(1, numel(temp_matrix)/const_insert)];
    temp_matrix = reshape(temp_matrix, temp_size(1)/const_insert*(const_insert+1) , temp_size(2));
    temp_matrix = temp_matrix(1:size(output_matrix,1) + ceil(size(output_matrix,1)/const_insert),:);
    %% insert column
    output_matrix = temp_matrix';

    temp_size = [ceil(size(output_matrix,1)/const_insert)*const_insert, size(output_matrix,2)] ;
    temp_matrix = -100 * ones(temp_size);
    temp_matrix(1:size(output_matrix,1),1:size(output_matrix,2)) = output_matrix;

    temp_matrix = reshape(temp_matrix, const_insert, numel(temp_matrix)/const_insert);
    temp_matrix = [temp_matrix; -100 * ones(1, numel(temp_matrix)/const_insert)];
    temp_matrix = reshape(temp_matrix, temp_size(1)/const_insert*(const_insert+1) , temp_size(2));
    temp_matrix = temp_matrix(1:size(output_matrix,1) + ceil(size(output_matrix,1)/const_insert),:);

    output_matrix = temp_matrix';
    %% display matrix
    figure(fig1);
    imshow(kron(output_matrix,ones(ceil(888 / (visible + hidden + 1)))), [-disp_scale, disp_scale], 'Colormap', [0,0,0;jet(256)]);
    saveas(gcf,'matrix.fig');
    %% visible
    const_insert = const_h;
    output_matrix = cpu_matrix;
    output_matrix = reshape(output_matrix(1:visible,visible+hidden+1), const_insert, visible / const_insert);

    %% insert column
    output_matrix = output_matrix';
    temp_size = [ceil(size(output_matrix,1)/const_insert)*const_insert, size(output_matrix,2)] ;
    temp_matrix = -100 * ones(temp_size);
    temp_matrix(1:size(output_matrix,1),1:size(output_matrix,2)) = output_matrix;

    temp_matrix = reshape(temp_matrix, const_insert, numel(temp_matrix)/const_insert);
    temp_matrix = [temp_matrix; -100 * ones(1, numel(temp_matrix)/const_insert)];
    temp_matrix = reshape(temp_matrix, temp_size(1)/const_insert*(const_insert+1) , temp_size(2));
    temp_matrix = temp_matrix(1:size(output_matrix,1) + ceil(size(output_matrix,1)/const_insert),:);
    output_matrix = temp_matrix';
    %% display visible units
    figure(fig2);
    imshow(kron(output_matrix, ones(10)), [-disp_scale, disp_scale], 'Colormap', [0,0,0;jet(256)]);
    saveas(gcf,'visible_bias.fig');
    %% hidden
    if hidden == visible
        const_insert = const_h;
        output_matrix = cpu_matrix;
        output_matrix = reshape(output_matrix(visible+1:visible+hidden,visible+hidden+1), const_insert, visible / const_insert);
        %% insert column
        output_matrix = output_matrix';
        temp_size = [ceil(size(output_matrix,1)/const_insert)*const_insert, size(output_matrix,2)];
        temp_matrix = -100 * ones(temp_size);
        temp_matrix(1:size(output_matrix,1),1:size(output_matrix,2)) = output_matrix;

        temp_matrix = reshape(temp_matrix, const_insert, numel(temp_matrix)/const_insert);
        temp_matrix = [temp_matrix; -100 * ones(1, numel(temp_matrix)/const_insert)];
        temp_matrix = reshape(temp_matrix, temp_size(1)/const_insert*(const_insert+1) , temp_size(2));
        temp_matrix = temp_matrix(1:size(output_matrix,1) + ceil(size(output_matrix,1)/const_insert),:);
        output_matrix = temp_matrix';
        %% display hidden units
        figure(fig3);
        imshow(kron(output_matrix, ones(10)), [-disp_scale, disp_scale], 'Colormap', [0,0,0;jet(256)]);
        saveas(gcf,'hidden_bias.fig');
    end
end