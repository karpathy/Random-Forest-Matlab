function model = forestTrain(X, Y, opts)
    % X is NxD, where rows are data points
    % for convenience, for now we assume X is 0 mean unit variance. If it
    % isn't, preprocess your data with
    %
    % X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
    %
    % If this condition isn't satisfied, some weak learners won't work out
    % of the box in current implementation.
    %
    % Y is discrete Nx1 vector of labels
    % model can be plugged into forestTest()
    %
    % decent default opts are:
    % opts.depth= 5;
    % opts.numTrees= 400;
    % opts.numSplits= 30;
    % opts.classifierID= 1:4
    %
    % which means use depth 5 trees, train 400 of them, use 30 random
    % splits when training each weak learner, and use a whole mixed bag of
    % weak learners for fun. (Ok usually may want to restrict a bit based 
    % on data)
    
    numTrees= 100;
    
    if nargin < 3, opts= struct; end
    if isfield(opts, 'numTrees'), numTrees= opts.numTrees; end

    treeModels= cell(1, numTrees);
    for i=1:numTrees
        treeModels{i} = treeTrain(X, Y, opts);
    end
    
    model.treeModels = treeModels;
end