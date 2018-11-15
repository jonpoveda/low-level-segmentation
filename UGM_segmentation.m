clear all;
close all;
clc;

% im_name='3_12_s.bmp';
% im_name='2_1_s.bmp';
im_name='7_9_s.bmp';
toLab = 0;

% TODO: Update library path
% Add  library paths
basedir='UGM/';
addpath(genpath(basedir));

%Set model parameters
%cluster coloryaou
K=4; % Number of color clusters (=number of states of hidden variables)
nStates = K;
%Pair-wise parameters
smooth_term=[-10 10]; % Potts Model

%Load images
im = imread(im_name);

NumFils = size(im,1);
NumCols = size(im,2);

%Convert to LAB colors space
if toLab == 1
  im = RGB2Lab(im);
end

%Preparing data for GMM fiting
%
% TODO: define the unary energy term: data_term
% nodePot = P( color at pixel 'x' | Cluster color 'c' )  
% im = rgb2gray(im);
% imshow(im);
im = double(im);
x = reshape(im, [size(im,1)*size(im,2), size(im,3)]);
gmm_color = gmdistribution.fit(x,K);
% In case the covariance matrix is ill composed
% gmm_color = gmdistribution.fit(x,K, 'CovarianceType', 'diagonal', ...
%   'SharedCovariance', true);

mu_color = gmm_color.mu;

data_term = gmm_color.posterior(x);
nodePot = data_term;  % According to the definition of 'nodePot' above
% Most probable neighbour 'c'
[~, c] = max(data_term,[],2);

nNodes = NumFils*NumCols;  % Each pixel is a node
%nStates = 4; % 4-neighbourhood (equal to K (always??)

% Standardize Features
Xstd = UGM_standardizeCols(reshape(x,[1 3 nNodes]),1);
size(Xstd)

% nodePot=[];
% nodePot = zeros(nNodes,nStates);
% % r
% nodePot(:,1,1) = exp(-1-2.5*Xstd(1,1,:));
% nodePot(:,1,2) = 1;
% % g
% nodePot(:,2,1) = exp(-1-2.5*Xstd(1,2,:));
% nodePot(:,2,2) = 1;
% % b
% nodePot(:,3,1) = exp(-1-2.5*Xstd(1,3,:));
% nodePot(:,3,2) = 1;
%Building 4-grid
%Build UGM Model for 4-connected segmentation
disp('create UGM model');

% Create UGM data
[edgePot,edgeStruct] = CreateGridUGMModel(NumFils, NumCols, nStates,...
  smooth_term);

if ~isempty(edgePot)

    % Color clustering (w/o graphical models)
    % TODO: find out why there was a minimum applied to a probability
    %[~,c] = min(reshape(data_term,[NumFils*NumCols K]),[],2);
    disp('Colour clustering (w/o GM)');
    tic;
    [~,c] = max(reshape(data_term,[NumFils*NumCols K]),[],2);
    im_c= reshape(mu_color(c,:),size(im));
    toc;
    
    % Call different UGM inference algorithms
    % Loopy Belief Propagation (inference)
    disp('Loopy Belief Propagation'); 
    tic;
    edgeStruct.maxIter = int32(200);
%     [nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
%     [~, c_lbp] = max(nodeBelLBP,[],2);
%     toc;
% Equivalent to the above but slightly faster (thanks to mex)
    maxOfMarginalsLBPdecode = UGM_Decode_MaxOfMarginals(nodePot,edgePot,...
      edgeStruct,@UGM_Infer_LBP);
    im_lbp = reshape(mu_color(maxOfMarginalsLBPdecode,:), size(im));
    toc;
    
    % Max-sum (decoding)
    disp('Max-sum (decoding)'); 
    tic;
    % Modify default maximum number of iterations
    edgeStruct.maxIter = int32(200);
    decodeLBP = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);
    im_bp= reshape(mu_color(decodeLBP,:),size(im));
    toc;
    
    % TODO: apply other inference algorithms and compare their performance
    %
    % Tree-reweighted Loopy Belief Propagation (inference)
    % Super slow!
%     tic;
%     edgeStruct.maxIter = 100;
%     [nodeBelTRBP,edgeBelTRBP,logZTRBP] = UGM_Infer_TRBP(nodePot,edgePot,edgeStruct);
%     toc;
%     [~, c_trbp] = max(nodeBelTRBP, [], 2);
%     im_trbp = reshape(mu_color(c_trbp, :), size(im));
    
    % - Graph Cut (cannot use it, we have nStates = K which is > 2)
    % Alpha-Beta Swap (decoding)
    disp('Alpha-Beta Swap (decoding)');
    tic;
    edgeStruct.maxIter = int32(200);
    alphaBetaDecode = UGM_Decode_AlphaBetaSwap(nodePot,edgePot,...
      edgeStruct,@UGM_Decode_GraphCut);
    im_abd = reshape(mu_color(alphaBetaDecode,:), size(im));
    toc;
    
    % Sampling for Inference (Gibbs Sampler + maximum of marginals)
    disp('Sampling for Inference (Gibbs Sampler + maximum of marginals)');
    tic;
    edgeStruct.maxIter = int32(200);
    burnIn=1000;  % Generate 'burnIn samples
    maxOfMarginalsGibbsDecode = UGM_Decode_MaxOfMarginals(nodePot,...
      edgePot,edgeStruct,@UGM_Infer_Sample,@UGM_Sample_Gibbs,burnIn);
    im_mmgd=reshape(mu_color(maxOfMarginalsGibbsDecode,:), size(im));
    toc;
    
    % - Linear Programing Relaxation

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