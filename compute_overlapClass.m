function [iou_class] = compute_overlapClass(res_class, gt_class,...
  NumRows, NumCols, NumChannels)
% Compute IoU (intersection over union) between estimated & ground truth
% class maps (or masks)

% Define mapping between classes (res_class => gt_class)
%   This happens because of the weakly labels and the order they are
%   encountered (column by column by 'unique')
%class_map = [2, 1, 4, 3];  % For 'tree' photo
%class_map = [2, 1, 4, 3];  % For 'car' photo
% Hard to evaluate because in the clustering grass & tree get merged
% And there is another 

gt_resize = reshape(gt_class, [NumRows * NumCols, NumChannels]);
res_resize = reshape(res_class, NumRows, NumCols);
classes = unique(gt_resize, 'rows');
num_classes = size(classes, 1);
gt_labels = rgb2ind(gt_class, num_classes);

iou_class = zeros(num_classes-1, 1);


% Compute IoU for all classes in the GT
for k = 1:num_classes-1 
  gt_mask = zeros(size(gt_labels));
  res_mask = zeros(size(gt_labels));
  
  gt_mask(gt_labels == k) = 1;
  %res_mask(res_resize == class_map(k)) = 1;
  res_mask(res_resize == k) = 1;

  intersection = res_mask & gt_mask;
  union = res_mask | gt_mask;
  
  % IoU = sum(pixels_in_intersection) / sum(pixels_in_union)
  iou_class(k) = sum(intersection(:)) / sum(union(:));
end