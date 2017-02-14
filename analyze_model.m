% 7 * 7 * 176
distances = [49,36,25,16,9,4,1,0];
% distance_mask = false(length(distances), const_h * const_w * const_c, const_h * const_w * const_c);
% for dis_idx = 1:length(distances)
%     dis = distances(dis_idx);
%     for x1 = 1:const_h
%         for y1 = 1:const_w
%             for c1 = 1:const_c
%                 for x2 = 1:const_h
%                     for y2 = 1:const_w
%                         for c2 = 1:const_c
%                             if (x1 - x2)^2 + (y1 - y2)^2 <= dis && ~(x1==x2 && y1 == y2 && c1 == c2)
%                                 distance_mask(dis_idx, hwc2x(x1,y1,c1,const_h, const_w), hwc2x(x2,y2,c2,const_h, const_w)) = 1;
%                             end
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end
load('distance_mask.mat');
masked_matrix = zeros(length(distances), const_h * const_w * const_c, const_h * const_w * const_c);

for dis_idx = 1:length(distances)
    masked_matrix(dis_idx, :, :) = gather(matrix(1:const_h * const_w * const_c,1:const_h * const_w * const_c));
end

X = -5e3:1e1:5e3;
H = zeros(length(distances), length(X));
for dis_idx = 1:length(distances)
    masked_matrix(dis_idx, :, :) = distance_mask(dis_idx, :, :) .* masked_matrix(dis_idx, :, :);
    H(dis_idx, :) = hist(masked_matrix(dis_idx, :), X);
end
H(:,501) = (H(:,500) + H(:,502)) / 2;
figure;
plot(X, H);
legend('7','6','5','4','3','2','1','0');