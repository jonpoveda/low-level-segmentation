function [edgePot,edgeStruct]=CreateGridUGMModel(im, nRows, nCols, K, pairwise_model, lambda)
%
% im: image to be segmented but reordered as a [nRows x nCols] x nChannels
% NumFils, NumCols: image dimensions
% K: number of states
% lambda: smoothing factor (Potts and Gaussian Edge Potentials)

tic;
nNodes = nRows * nCols;

%% Construct adjacency matrix
adj = sparse(nNodes, nNodes);

% Add Down Edges
ind = 1:nNodes;
% No Down edge for last row
exclude = sub2ind([nRows, nCols], repmat(nRows, [1, nCols]), 1:nCols);
ind = setdiff(ind,exclude);
adj(sub2ind([nNodes, nNodes], ind, ind + 1)) = 1;

% Add Right Edges
ind = 1:nNodes;
% No right edge for last column
exclude = sub2ind([nRows, nCols], 1:nRows, repmat(nCols, [1, nRows]));
ind = setdiff(ind, exclude);
adj(sub2ind([nNodes, nNodes], ind, ind + nRows)) = 1;

% Add Up/Left Edges
adj = adj + adj';

%% Initialise edgeStruct and edgePot
% Create edgeStruct and initialise with default parameters
edgeStruct = [];
edgeStruct = UGM_makeEdgeStruct(adj, K);

% Create edgePot and initialise it to 0
edgePot = [];
edgePot = zeros(K, K, edgeStruct.nEdges);

%% Initialise pairwise model
if pairwise_model == 1 || pairwise_model == 4
  % Potts model
  % pot same is a K*K matrix with lambda(1) diagonal and lambda(2) off-diagonal
  % e.g.: K=4  | l(1)  l(2) l(2)	l(2) |
  % potts    = | l(2)  l(1)	l(2)	l(2) |
  %            | l(2)  l(2)	l(1)  l(2) |
  %            | l(2)  l(2)	l(2)	l(1) |
  potts_potential = lambda(2) * ones(K,K);
  potts_potential(1:(K+1):end) = lambda(1);
  
  if pairwise_model == 4
    % Gaussian parwise potentials (from paper, see below)
    w_app =  2;         % Weight for the appearance kernel
    w_smooth =  1;      % Weight for the smoothness kernel
    theta_alpha = 61;   % Standard deviation for "Nearness" factor
    theta_beta = 11;    % Standard deviation for "Similarity" factor
    theta_gamma = 1;     % Standard deviation of the smoothness
    
  end
  
elseif pairwise_model == 2
  % Constants for Ising model (very important to try a few combinations)
  c1 = 1.8;
  c2 = 0.3;
  Xstd = UGM_standardizeCols(reshape(im,[1 3 nNodes]),1);
  
elseif pairwise_model == 3
  % Potts but with gaussian weighting
  sigma = 2;
  h = fspecial('gaussian', K, sigma);
  
end

%% Compute pairwise model
for e = 1:edgeStruct.nEdges
  
  switch pairwise_model
    case 1
      % Assign pre-computed Potts potential (see above)
      edgePot(:,:,e) = potts_potential;
      
    case 2
      % Ising model (theoretically worse than Potts...)
      n1 = edgeStruct.edgeEnds(e, 1);
      n2 = edgeStruct.edgeEnds(e, 2);
      ising_potential = exp(c1 + c2 * 1 / (1 + abs(Xstd(n1) - Xstd(n2))));
      edgePot(:, :, e) = ising_potential;
      
    case 3
      % Potts but with gaussian weighting
      potts_gauss = lambda(2) * ones(K, K);
      potts_gauss(1:(K+1):end) = lambda(1);
      edgePot(:, :, e) = h .* potts_gauss;
      
    case 4
      % Gaussian Edge Potentials from: 
      %   "Efficient Inference in Fully Connected CRFs with
      %    Gaussian Edge Potentials"
      
      % Get edge ends for current edge 'e'
      n1 = edgeStruct.edgeEnds(e,1);
      n2 = edgeStruct.edgeEnds(e,2);
      
      % Construct pairwise terms
      % Get pixel locations as (row,col) to discriminate close/far pixels
      [x1, y1] = ind2sub([nRows, nCols], n1);
      [x2, y2] = ind2sub([nRows, nCols], n2);
      
      % Compute distance between pixels
      dist_points = sqrt((x2 - x1)^2 + (y2 - y1)^2);
      % Compute "nearness" exponent
      nearness = abs(dist_points)^2 / 2 * theta_alpha^2;
      
      % Get colour for each edge:
      c1 = im(n1, :);
      c2 = im(n2, :);
      % Compute colour distance ('true' for Lab colourspace, else => RGB)
      colour_dist = distance_colours(c1, c2, true);
      similarity = abs(colour_dist)^2 / 2 * theta_beta^2;
      smoothness = abs(dist_points)^2 / 2 * theta_gamma^2;
      
      % Compute appearance term
      app_term = exp(-nearness - similarity);
      % Compute smoothness term
      smooth_term = exp(-smoothness);
      gauss_kernel = w_app * app_term + w_smooth * smooth_term;
      gauss_edge_potential = potts_potential * gauss_kernel;
      
      % Assign potential to current edge
      edgePot(:, :, e) = gauss_edge_potential;
      
  end
end
toc;