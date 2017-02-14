function x = hwc2x(h, w, c, const_h, const_w)
    x = (c-1) * const_h * const_w + (w-1) * const_h + h;
end