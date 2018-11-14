function [edgePot,edgeStruct]=CreateGridUGMModel(nRows, nCols, K, lambda)
%
%
% NumFils, NumCols: image dimension
% K: number of states
% lambda: smoothing factor



tic

nNodes = nRows*nCols;
 
adj = sparse(nNodes,nNodes);
 
% Add Down Edges
ind = 1:nNodes;
exclude = sub2ind([nRows nCols],repmat(nRows,[1 nCols]),1:nCols); % No Down edge for last row
ind = setdiff(ind,exclude);
adj(sub2ind([nNodes nNodes],ind,ind+1)) = 1;
 
% Add Right Edges
ind = 1:nNodes;
exclude = sub2ind([nRows nCols],1:nRows,repmat(nCols,[1 nRows])); % No right edge for last column
ind = setdiff(ind,exclude);
adj(sub2ind([nNodes nNodes],ind,ind+nRows)) = 1;
 
% Add Up/Left Edges
adj = adj+adj';

edgeStruct=[];
edgeStruct = UGM_makeEdgeStruct(adj,K);

% Standardize Features
% Xstd = UGM_standardizeCols(reshape(X,[1 1 nNodes]),1);

edgePot=[];
edgePot = zeros(K,K,edgeStruct.nEdges);
for e = 1:edgeStruct.nEdges
   n1 = edgeStruct.edgeEnds(e,1);
   n2 = edgeStruct.edgeEnds(e,2);

   % pot same is a K*K matrix with 0 diagonal and lambda(2) off-diagonal
   % e.g.: K=4  | 0     lmbda  lmbda lmbda |
   % pot_same = | lmbda   0    lmbda lmbda |
   %            | lmbda lmbda   0    lmbda |
   %            | lmbda lmbda  lmbda   0   |
   pot_same = lambda(2) * ones(K,K);
   pot_same(1:(K+1):end) = 0;
   %pot_same = exp(1.8 + .3*1/(1+abs(Xstd(n1)-Xstd(n2))));
   edgePot(:,:,e) = pot_same; %[pot_same 1;1 pot_same];
end


toc;