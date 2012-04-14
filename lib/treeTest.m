function [Yhard, Ysoft] = treeTest(model, X, opts)
% Test a tree
% X is NxD, each D-dimensional row is a data point
% model comes from treeTrain()
% Yhard are hard assignments to X's, Ysoft is NxK array of
% probabilities, where there are K classes.
    
if nargin < 3, opts= struct; end

d= model.depth;

[N, D]= size(X);
nd= 2^d - 1;
numInternals = (nd+1)/2 - 1;
numLeafs= (nd+1)/2;

Yhard= zeros(N, 1);
u= model.classes;
if nargout>1, Ysoft= zeros(N, length(u)); end

% if we can afford to store as non-sparse (100MB array, say), it is
% slightly faster.
if storage([N nd]) < 100 
    dataix= zeros(N, nd); % boolean indicator of data at each node
else
    dataix= sparse(N, nd); 
end

% Propagate data down the tree using weak classifiers at each node
for n = 1: numInternals
    
    % get relevant data at this node
    if n==1 
        reld = ones(N, 1)==1;
        Xrel= X;
    else
        reld = dataix(:, n)==1;
        Xrel = X(reld, :);
    end
    if size(Xrel,1)==0, continue; end % empty branch, ah well
    
    yhat= weakTest(model.weakModels{n}, Xrel, opts);
    
    dataix(reld, 2*n)= yhat;
    dataix(reld, 2*n+1)= 1 - yhat; % since yhat is in {0,1} and double
end

% Go over leafs and assign class probabilities
for n= (nd+1)/2 : nd
    ff= find(dataix(:, n)==1);
    
    hc= model.leafdist(n - (nd+1)/2 + 1, :);
    vm= max(hc);
    miopt= find(hc==vm);
    mi= miopt(randi(length(miopt), 1)); %choose a class arbitrarily if ties exist
    Yhard(ff)= u(mi);
    
    if nargout > 1
        Ysoft(ff, :)= repmat(hc, length(ff), 1);
    end
end
