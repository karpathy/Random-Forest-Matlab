function model = forestTrain(X, Y, opts)
    % X is NxD, where rows are data points
    % for convenience, for now we assume X is 0 mean unit variance. If it
    % isn't, preprocess your data with
    %
    % X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X) + 1e-10);
    %
    % If this condition isn't satisfied, some weak learners won't work out
    % of the box in current implementation.
    %
    % Y is discrete Nx1 vector of labels
    % model can be plugged into forestTest()
    %
    % decent default opts are:
    % opts.depth= 9;
    % opts.numTrees= 100; %(but more is _ALWAYS_ better, monotonically, no exceptions)
    % opts.numSplits= 5;
    % opts.classifierID= 2
    % opts.classifierCommitFirst= true;
    %
    % which means use depth 9 trees, train 100 of them, use 5 random
    % splits when training each weak learner. The last option controls
    % whether each node commits to a weak learner at random and then trains
    % it best it can, or whether each node tries all weak learners and
    % takes the one that does best. Empirically, it appears leaving this as
    % true (default) gives slightly better looking results.
    %
    
    numTrees= 100;
    verbose= false;
    
    if nargin < 3, opts= struct; end
    if isfield(opts, 'numTrees'), numTrees= opts.numTrees; end
    if isfield(opts, 'verbose'), verbose= opts.verbose; end

    treeModels= cell(1, numTrees);
    for i=1:numTrees
        
        treeModels{i} = treeTrain(X, Y, opts);
        
        % print info if verbose
        if verbose
            p10= floor(numTrees/10);
            if mod(i, p10)==0 || i==1 || i==numTrees
                fprintf('Training tree %d/%d...\n', i, numTrees);
            end
        end
    end
    
    model.treeModels = treeModels;
end