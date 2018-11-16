function [col_dist] = distance_colours(c1, c2, lab)
% Computes the distance between two colours in the RGB space
% The computation can be done in the RGB or Lab space.
% For perceptual reasons, it is strongly suggested to use the Lab space
% (default)

if lab
  % Convert RGB colours into Lab space
  c1_lab = rgb2lab(c1);
  c2_lab = rgb2lab(c2);
  
  % Compute mean
  % Compute distance in the Lab space (i.e.: DeltaE distance)
  % Note: we used the simplest, less accurate from CIE 1976 where we simply
  % compute the sqrt of the sum of differences per channels.
  col_dist = sqrt((c1_lab(1) - c2_lab(1))^2 + (c1_lab(2) - c2_lab(2))^2 +...
    (c1_lab(3) - c2_lab(3))^2);
  
else % RGB
  % Simply compute sqrt((diff)^2)
  col_dist = sqrt((c1(1) - c2(1))^2 + (c1(2) - c2(2))^2 +(c1(3) - c2(3))^2);
  
end
