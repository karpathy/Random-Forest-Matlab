function model = weakTrain(X, Y, opts)
% weak random learner
% can currently train:
% 1. decision stump: look along random dimension of data, choose threshold
% that maximizes information gain in class labels
% 2. 2D linear decision learner: same as decision stump but in 2D. I know,
% in general this could all be folded into single linear stump, but I make
% distinction for historical, cultural, and efficiency reasons.
% 3. Conic section learner: second order learning in 2D. i.e. x*y is a
% feature in addition to x, y and offset (as in 2.)
% 4. Distance learner. Picks a data point in train set and a threshold. The
% label is computed based on distance to the data point

classifierID= 1; % by default use decision stumps
numSplits= 30; 

if nargin < 3, opts = struct; end
if isfield(opts, 'classifierID')
    classifierID = opts.classifierID; 
    if length(classifierID) > 1
        % we were passed a list, sample a random weak learner from it for
        % this node
        classifierID= classifierID(randi(length(classifierID)));
    end
end
if isfield(opts, 'numSplits'), numSplits = opts.numSplits; end
    
model= struct;
model.classifierID= classifierID;
u= unique(Y);

[N, D]= size(X);

if classifierID == 1
    % Decision stump
        
    % random splitting dimension
    rd= randi(D);
    model.ix= rd;
    
    if size(X,1) <= 1
        % edge case: no data or 1 data point. Not much can be done
        model.t= 0;
        return;
    end
    
    % proceed to pick optimal splitting value t, based on Information Gain
    col= X(:, rd);
    tmin= min(col);
    tmax= max(col);
    Is= zeros(numSplits, 1);
    ts= zeros(numSplits, 1);
    i= 0;
    for t = linspace(tmin, tmax, numSplits)
        i=i+1;

        dec= col < t;

        % calculate information gain from dec
        YL= Y(dec);
        YR= Y(~dec);
        H= classEntropy(Y, u);
        HL= classEntropy(YL, u);
        HR= classEntropy(YR, u);
        Igain= H - length(YL)/length(Y)*HL - length(YR)/length(Y)*HR;

        Is(i)= Igain;
        ts(i)= t;
    end
    
    % choose the exact split spot randomly if there are several equivalent
    vopt= max(Is);
    ixopt= find(Is==vopt);
    iopt= ixopt(randi(length(ixopt), 1));
    model.t= ts(iopt);
    
elseif classifierID == 2
    % Linear classifier using 2 dimensions
    
    if size(X,1) <= 1
        % edge case: no data or 1 data point. Not much can be done
        model.r1= randi(D);
        model.r2= randi(D);
        model.w= randn(3, 1);
        return;
    end
    
    % Repeat some number of times: 
    % pick two dimensions, pick 3 random parameters, and see what happens
    maxgain= -1;
    for q= 1:numSplits
        
        r1= randi(D);
        r2= randi(D);
        
        % weigh our random parameter proposals according to variance in
        % each dimension
        w= randn(3, 1);
        dec= [X(:, [r1 r2]), ones(N, 1)]*w < 0;
        
        YL= Y(dec);
        YR= Y(~dec);
        H= classEntropy(Y, u);
        HL= classEntropy(YL, u);
        HR= classEntropy(YR, u);
        Igain= H - length(YL)/length(Y)*HL - length(YR)/length(Y)*HR;
        
        if Igain>maxgain
            maxgain = Igain;
            model.r1= r1;
            model.r2= r2;
            model.w= w;
        end
    end
    
elseif classifierID == 3
    % Conic section weak learner in 2D (not too good presently, what is the
    % best way to randomly suggest good parameters?
    
    if size(X,1) <= 1
        % edge case: no data or 1 data point. Not much can be done
        model.r1= randi(D);
        model.r2= randi(D);
        model.w= randn(4, 1);
        return;
    end
    
    % Repeat some number of times: 
    % pick two dimensions, pick 3 random parameters, and see what happens
    maxgain= -1;
    for q= 1:numSplits
        
        r1= randi(D);
        r2= randi(D);
        
        % weigh our random parameter proposals according to variance in
        % each dimension
        w= randn(4, 1);
        dec= [X(:, r1).*X(:,r2), X(:, [r1 r2]), ones(N, 1)]*w < 0;
        
        YL= Y(dec);
        YR= Y(~dec);
        H= classEntropy(Y, u);
        HL= classEntropy(YL, u);
        HR= classEntropy(YR, u);
        Igain= H - length(YL)/length(Y)*HL - length(YR)/length(Y)*HR;
        
        if Igain>maxgain
            maxgain = Igain;
            model.r1= r1;
            model.r2= r2;
            model.w= w;
        end
    end
    
elseif classifierID==4
    % RBF weak learner: Picks an example and bases decision on distance
    % threshold
    
    if size(X,1) <= 1
        % edge case: no data or 1 data point. Not much can be done
        model.x= zeros(size(X,2), 1);
        model.t= 0;
        return;
    end
    
    % Repeat some number of times: 
    % pick two dimensions, pick 3 random parameters, and see what happens
    maxgain= -1;
    for q= 1:numSplits
        
        % this is expensive, lets only recompute every once in a while...
        if mod(q-1,5)==0
            x= X(randi(size(X, 1)), :);
            dsts= pdist2(X, x);
        end
        
        t= rand()*(max(dsts)-min(dsts))+ min(dsts);
        dec= dsts < t;
        
        YL= Y(dec);
        YR= Y(~dec);
        H= classEntropy(Y, u);
        HL= classEntropy(YL, u);
        HR= classEntropy(YR, u);
        Igain= H - length(YL)/length(Y)*HL - length(YR)/length(Y)*HR;
        
        if Igain>maxgain
            maxgain = Igain;
            model.x= x;
            model.t= t;
        end
    end
    
else
    fprintf('Error in weak train! Classifier with ID = %d does not exist.\n', classifierID);
end

end

% Helper function for class entropy used with Decision Stump
function H= classEntropy(y, u)

    cdist= histc(y, u);
    cdist= cdist .* log(cdist);
    cdist(isnan(cdist))= 0;
    H= sum(cdist);
end
