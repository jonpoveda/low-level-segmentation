clear;
close all;
clc;

% Add  library paths
basedir = 'UGM/';
addpath(genpath(basedir));

%% Load image & its (weakly labelled) GT mask
% First image (house) => 2 classes + unlabelled class
% im_name = '3_12_s.bmp';
% gt = '3_12_s_GT.bmp';
% im_name='2_1_s.bmp';
% gt='2_1_s_GT.bmp';
im_name='7_9_s.bmp';
gt = '7_9_s_GT.bmp';
toLab = 0;

im = imread(im_name);
gt = imread(gt);
NumRows = size(im, 1);
NumCols = size(im, 2);
NumChannels = size(im, 3);

% Convert to LAB colors space
if toLab == 1
  im = RGB2Lab(im);
end

%% Set model parameters
%eval_results = false; % For all cases where K is different num_clusters
eval_results = true;  % Only when gt classes = num_clusters!
K = 5;    % Number of color clusters (=number of states of hidden variables)
nStates = K;
nNodes = NumRows * NumCols;  % Each pixel is a node

% Pair-wise parameters
pairwise_model = 2;   % Pairwise model to use (1=> Potts, 2==> Ising,)
smooth_term = [-10, 10]; % Potts Model

%% Define the unary energy term: data_term
% nodePot = P( color at pixel 'x' | Cluster color 'c' )
% im = rgb2gray(im);
% imshow(im);
im = double(im);
x = reshape(im, [NumRows * NumCols, NumChannels]);
%gmm_color = gmdistribution.fit(x, K);
% gmm_color = fitgmdist(x, K);

% --- WARNING ---
% In case the covariance matrix is ill composed
gmm_color = fitgmdist(x, K, 'CovarianceType', 'diagonal');%, ...
%    'SharedCovariance', true);

mu_color = gmm_color.mu;

data_term = gmm_color.posterior(x);
nodePot = data_term;  % According to the definition of 'nodePot' above

% Most probable neighbour 'c'
[~, c] = max(data_term, [], 2); % Not used here (recomputed below)=> remove


%% Build UGM Model for 4-connected segmentation
% Building 4-grid
disp('create UGM model');

% Create UGM data
[edgePot,edgeStruct] = CreateGridUGMModel(x, NumRows, NumCols, nStates,...
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
  [~,c] = max(reshape(data_term, [NumRows * NumCols, K]), [], 2);
  if (eval_results)
    [iou_clusterGMM] = compute_overlapClass(c, gt, NumRows, NumCols,...
      NumChannels);
    fprintf('IoU for clustering without GM (only GMM):\n');
    for k = 1:K
      fprintf('\t IoU class %d:  %.2f\n', k, iou_clusterGMM(k));
    end
  end
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
  
  if (eval_results)
    [iou_lbp_infer] = compute_overlapClass(maxOfMarginalsLBPdecode,...
      gt, NumRows, NumCols, NumChannels);
    fprintf('IoU for LBP infer\n');
    for k = 1:K
      fprintf('\t IoU class %d:  %.2f\n', k, iou_lbp_infer(k));
    end
    
  end
  
  im_lbp = reshape(mu_color(maxOfMarginalsLBPdecode, :), size(im));
  toc;
  
  % - Max-sum (decoding)
  %   Source: https://www.cs.ubc.ca/~schmidtm/Software/UGM/variational.html
  disp('Max-sum (decoding)');
  tic;
  % Maximum number of iterations
  edgeStruct.maxIter = int32(200);
  decodeLBP = UGM_Decode_LBP(nodePot, edgePot, edgeStruct);
  if (eval_results)
    [iou_lbp_decode] = compute_overlapClass(decodeLBP, gt, NumRows,...
      NumCols, NumChannels);
    fprintf('IoU for LBP decode:\n');
    for k = 1:K
      fprintf('\t IoU class %d:  %.2f\n', k, iou_lbp_decode(k));
    end
  end
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
  
  if (eval_results)
    [iou_alphaBeta] = compute_overlapClass(alphaBetaDecode, gt, NumRows,...
      NumCols, NumChannels);
    fprintf('IoU for Alpha Beta:\n');
    for k = 1:K
      fprintf('\t IoU class %d:  %.2f\n', k, iou_alphaBeta(k));
    end
  end
  
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
  if (eval_results)
    [iou_gibbsDecode] = compute_overlapClass(maxOfMarginalsGibbsDecode,...
      gt, NumRows, NumCols, NumChannels);
    fprintf('IoU for Gibbs Decode:\n');
    for k = 1:K
      fprintf('\t IoU class %d:  %.2f\n', k, iou_gibbsDecode(k));
    end
  end
  
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