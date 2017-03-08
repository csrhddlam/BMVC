function [h, w, c] = x2hwc(x, const_h, const_w, const_c)
    x = x - 1;
    h = mod(floor(x / 1), const_h) + 1;
    w = mod(floor(x / const_h), const_w) + 1;
    c = mod(floor(x / const_h / const_w), const_c) + 1;
end