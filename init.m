% delta_free_energy = zeros(1, const_iteration);
% fig_history = figure;
% fig1 = figure;
% fig2 = figuremat_constents
% fig3 = figure;

%% matrix init

matrix = randn(visible + hidden + 1, visible + hidden + 1) * init_var;
matrix = 0.5 * matrix + 0.5 * matrix'; % symmetry

mask = true(visible + hidden + 1, visible + hidden + 1);
for temp = 1:visible + hidden + 1
    matrix(temp, temp) = 0;
    mask(temp, temp) = 0;
end

if type ==0 % Type 0: General BM
elseif type == 1 % Type 1: no visible-visible
    matrix(1:visible,1:visible) = 0;
    mask(1:visible,1:visible) = 0;
elseif type == 2 % Type 2: only visible-visible units
    matrix(1:visible + hidden + 1, visible + 1: visible + hidden) = 0;
    mask(1:visible + hidden + 1, visible + 1: visible + hidden) = 0;
    matrix(visible + 1: visible + hidden, 1:visible + hidden + 1) = 0;
    mask(visible + 1: visible + hidden, 1:visible + hidden + 1) = 0;
elseif type == 3 % Type 3: Restricted BM
    matrix(1:visible,1:visible) = 0;
    mask(1:visible,1:visible) = 0;
    matrix(visible + 1: visible + hidden, visible + 1: visible + hidden) = 0;
    mask(visible + 1: visible + hidden, visible + 1: visible + hidden) = 0;
elseif type == 4 && hidden == visible % Type 4: Wenhao
    matrix(1:visible,1:visible) = 0;
    mask(1:visible,1:visible) = 0;
    matrix(1: hidden, visible + 1: visible + hidden) = eye(hidden) * fix_w12;
    mask(1: hidden, visible + 1: visible + hidden) = zeros(hidden);
    matrix(visible + 1: visible + hidden, 1: hidden) = eye(hidden) * fix_w12;
    mask(visible + 1: visible + hidden, 1: hidden) = zeros(hidden);
    % matrix(visible + 1: visible + hidden, visible + 1: visible + hidden) = matrix_init;
elseif type == 5 % nothing but bias
    matrix(1: visible + hidden, 1:visible + hidden) = 0;
    mask(1: visible + hidden, 1:visible + hidden) = 0;
    matrix(visible + 1: visible + hidden, 1:visible + hidden + 1) = 0;
    mask(visible + 1: visible + hidden, 1:visible + hidden + 1) = 0;
end

% load('distance_mask.mat');
% mask(1:visible,1:visible) = reshape(distance_mask(7,:,:),visible, visible) .* mask(1:visible,1:visible);
% matrix = matrix .* mask;

matrix = gpuArray(single(matrix));
mask = gpuArray(mask);