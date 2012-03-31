function yhat = weakTest(model, X, opts)
% X is NxD as usual. 
% see weakTrain for more info.

if nargin < 3, opts = struct; end

[N, D]= size(X);

if model.classifierID== 1
    
    % Decision stump classifier
    yhat = double(X(:, model.ix) < model.t);

elseif model.classifierID== 2
    
    % 2-D linear clussifier stump
    yhat = double([X(:, [model.r1 model.r2]), ones(N, 1)]*model.w < 0);
    
elseif model.classifierID== 3
    
    % 2-D conic section learner
    r1= model.r1;
    r2= model.r2;
    yhat= double([X(:, r1).*X(:,r2), X(:, [r1 r2]), ones(N, 1)]*model.w < 0);
    
elseif model.classifierID== 4
    
    % RBF, distance based learner
    yhat= double(pdist2(X, model.x) < model.t);
    
else
    fprintf('Error in weak test! Classifier with ID = %d does not exist.\n', classifierID);
end


end