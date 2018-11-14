clear all;
close all;
clc;

im_name='3_12_s.bmp';

% TODO: Update library path
% Add  library paths
basedir='UGM/';
addpath(genpath(basedir));

%Set model parameters
%cluster coloryaou
K=8; % Number of color clusters (=number of states of hidden variables)

%Pair-wise parameters
smooth_term=[0.2 100]; % Potts Model

%Load images
im = imread(im_name);

NumFils = size(im,1);
NumCols = size(im,2);

%Convert to LAB colors space
% TODO: Uncomment if you want to work in the LAB space
%
% im = RGB2Lab(im);



%Preparing data for GMM fiting
%
% TODO: define the unary energy term: data_term
% nodePot = P( color at pixel 'x' | Cluster color 'c' )  
% im = rgb2gray(im);
% imshow(im);
im = double(im);
x = reshape(im, [size(im,1)*size(im,2), size(im,3)]);
gmm_color = gmdistribution.fit(x,K);
mu_color = gmm_color.mu;

data_term = gmm_color.posterior(x);
nodePot = data_term;  % According to the definition of 'nodePot' above
% Most probable neighbour 'c'
[~, c] = max(data_term,[],2);


nNodes = NumFils*NumCols;  % Each pixel is a node
nStates = 8; % 4-neighbourhood (equal to K (always??)

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
[edgePot,edgeStruct] = CreateGridUGMModel(NumFils, NumCols, nStates, smooth_term);


if ~isempty(edgePot)

    % color clustering
    [~,c] = min(reshape(data_term,[NumFils*NumCols K]),[],2);
    im_c= reshape(mu_color(c,:),size(im));
    
    % Call different UGM inference algorithms
    disp('Loopy Belief Propagation'); tic;
    [nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);toc;
    [~, c_lbp] = max(nodeBelLBP,[],2);
    % Need to convert im_lbp to image dimensions 
    im_lbp = reshape(mu_color(c_lbp,:), size(im));
    
    % Max-sum
    disp('Max-sum'); tic;
    decodeLBP = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);
    im_bp= reshape(mu_color(decodeLBP,:),size(im));
    toc;
    
    
    % TODO: apply other inference algorithms and compare their performance
    %
    % - Graph Cut
    % - Linear Programing Relaxation
    
    figure
    % If we converted to Lab
%     subplot(2,2,1),imshow(Lab2RGB(im));xlabel('Original');
%     subplot(2,2,2),imshow(Lab2RGB(im_c),[]);xlabel('Clustering without GM');
%     subplot(2,2,3),imshow(Lab2RGB(im_bp),[]);xlabel('Max-Sum');
%     subplot(2,2,4),imshow(Lab2RGB(im_lbp),[]);xlabel('Loopy Belief Propagation');
    
    % If we used RGB
    subplot(2,2,1),imshow(uint8(im));xlabel('Original');
    subplot(2,2,2),imshow(uint8(im_c),[]);xlabel('Clustering without GM');
    subplot(2,2,3),imshow(uint8(im_bp),[]);xlabel('Max-Sum');
    subplot(2,2,4),imshow(uint8(im_lbp),[]);xlabel('Loopy Belief Propagation');
    
else
   
    error('You have to implement the CreateGridUGMModel.m function');

end