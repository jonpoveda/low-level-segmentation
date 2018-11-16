clear;
close all;
clc;

% Add  library paths
basedir='UGM/';
addpath(genpath(basedir));

%% Load image
im_name='3_12_s.bmp';
% im_name='2_1_s.bmp';
% im_name='7_9_s.bmp';
toLab = 0;

% Load images
im = imread(im_name);
NumFils = size(im, 1);
NumCols = size(im, 2);

% Convert to LAB colors space
if toLab == 1
  im = RGB2Lab(im);
end

%% Set model parameters
K = 4;    % Number of color clusters (=number of states of hidden variables)
nStates = K;
nNodes = NumFils * NumCols;  % Each pixel is a node

% Pair-wise parameters
pairwise_model = 4;   % Pairwise model to use (1=> Potts, 2==> Ising,)

if pairwise_model == 1 || pairwise_model == 4
  smooth_term=[-10 10]; % Potts Model
end


%% Define the unary energy term: data_term
% nodePot = P( color at pixel 'x' | Cluster color 'c' )
% im = rgb2gray(im);
% imshow(im);
im = double(im);
x = reshape(im, [size(im, 1) * size(im, 2), size(im, 3)]);
gmm_color = gmdistribution.fit(x, K);

% --- WARNING ---
% In case the covariance matrix is ill composed
% gmm_color = gmdistribution.fit(x,K, 'CovarianceType', 'diagonal', ...
%   'SharedCovariance', true);

mu_color = gmm_color.mu;

data_term = gmm_color.posterior(x);
nodePot = data_term;  % According to the definition of 'nodePot' above

% Most probable neighbour 'c'
[~, c] = max(data_term, [], 2); % Not used here (recomputed below)=> remove


%% Build UGM Model for 4-connected segmentation
% Building 4-grid
disp('create UGM model');

% Create UGM data
[edgePot,edgeStruct] = CreateGridUGMModel(x, NumFils, NumCols, nStates,...
  pairwise_model, smooth_term);

%% Run and compare different inference algorithms
if ~isempty(edgePot)
  
  % Color clustering (w/o graphical models)
  % TODO: find out why there was a minimum applied to a probability
  %[~,c] = min(reshape(data_term,[NumFils*NumCols K]),[],2);
  disp('Colour clustering (w/o GM)');
  tic;
  % Simply compute maximum of a posteriori probability of pixel 'x'
  % belonging to cluster 'c'
  [~,c] = max(reshape(data_term, [NumFils * NumCols, K]), [], 2);
  % Use mean colour of most likely cluster
  im_c = reshape(mu_color(c, :), size(im));
  toc;
  
  % Different UGM inference algorithms
  
  % - Loopy Belief Propagation (inference)
  %   Source: https://www.cs.ubc.ca/~schmidtm/Software/UGM/variational.html
  disp('Loopy Belief Propagation');
  tic;
  edgeStruct.maxIter = int32(200);
  %     [nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
  %     [~, c_lbp] = max(nodeBelLBP,[],2);
  %     toc;
  % Equivalent to the above but slightly faster (thanks to mex end-to-end)
  
  maxOfMarginalsLBPdecode = UGM_Decode_MaxOfMarginals(nodePot, edgePot,...
    edgeStruct, @UGM_Infer_LBP);
  im_lbp = reshape(mu_color(maxOfMarginalsLBPdecode, :), size(im));
  toc;
  
  % - Max-sum (decoding)
  %   Source: https://www.cs.ubc.ca/~schmidtm/Software/UGM/variational.html
  disp('Max-sum (decoding)');
  tic;
  % Maximum number of iterations
  edgeStruct.maxIter = int32(200);
  decodeLBP = UGM_Decode_LBP(nodePot, edgePot, edgeStruct);
  im_bp = reshape(mu_color(decodeLBP, :), size(im));
  toc;
  
  % TODO: apply other inference algorithms and compare their performance
  
  % - Graph Cut (cannot use it, we have nStates = K which is > 2)
  
  % - Alpha-Beta Swap (decoding)
  %   Source: https://www.cs.ubc.ca/~schmidtm/Software/UGM/alphaBeta.html
  disp('Alpha-Beta Swap (decoding)');
  tic;
  edgeStruct.maxIter = int32(200);
  alphaBetaDecode = UGM_Decode_AlphaBetaSwap(nodePot, edgePot,...
    edgeStruct, @UGM_Decode_GraphCut);
  im_abd = reshape(mu_color(alphaBetaDecode, :), size(im));
  toc;
  
  % - Sampling for Inference (Gibbs Sampler + maximum of marginals)
  %   Source: https://www.cs.ubc.ca/~schmidtm/Software/UGM/MCMC.html
  disp('Sampling for Inference (Gibbs Sampler + maximum of marginals)');
  tic;
  edgeStruct.maxIter = int32(200);
  burnIn = 1000;      % Generate burnIn samples
  maxOfMarginalsGibbsDecode = UGM_Decode_MaxOfMarginals(nodePot,...
    edgePot, edgeStruct, @UGM_Infer_Sample, @UGM_Sample_Gibbs, burnIn);
  im_mmgd = reshape(mu_color(maxOfMarginalsGibbsDecode, :), size(im));
  toc;
  
  % - Linear Programing Relaxation (super slow and works with very small
  % images in a reasonable amount of time)
  
  % Convert back to RGB (if needed)
  if (toLab == 1)
    im = Lab2RGB(im);
    im_c = Lab2RGB(im_c);
    im_lbp = Lab2RGB(im_lbp);
    im_bp = Lab2RGB(im_bp);
    im_abd = Lab2RGB(im_abd);
    im_mmgd = Lab2RGB(im_mmgd);
  end
  
  figure('Name', 'Comparison of inference algorithms');
  
  subplot(2,3,1),imshow(uint8(im));
  xlabel('Original');
  
  subplot(2,3,2),imshow(uint8(im_c),[]);
  xlabel('Clustering without GM');
  
  subplot(2,3,3),imshow(uint8(im_bp),[]);
  xlabel('Max-Sum (decoding)');
  
  subplot(2,3,4),imshow(uint8(im_lbp),[]);
  xlabel('Loopy Belief Propagation');
  
  subplot(2,3,5),imshow(uint8(im_abd),[]);
  xlabel('Alpha-Beta Swap (decoding)');
  
  subplot(2,3,6),imshow(uint8(im_mmgd),[]);
  xlabel('Max-of-marginals (GibbsSample)');
  
else
  
  error('You have to implement the CreateGridUGMModel.m function');
  
end