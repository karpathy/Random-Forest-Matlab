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

classifierID= 1; % by default use decision stumps only
numSplits= 30; 
classifierCommitFirst= true;

if nargin < 3, opts = struct; end
if isfield(opts, 'classifierID'), classifierID = opts.classifierID; end
if isfield(opts, 'numSplits'), numSplits = opts.numSplits; end
if isfield(opts, 'classifierCommitFirst'), classifierCommitFirst = opts.classifierCommitFirst; end

if classifierCommitFirst
    % commit to a weak learner first, then optimize its parameters only. In
    % this variation, different weak learners don't compete for a node.
    if length(classifierID)>1
        classifierID= classifierID(randi(length(classifierID)));
    end
end

u= unique(Y);
[N, D]= size(X);

if N == 0
    % edge case. No data reached this leaf. Don't do anything...
    model.classifierID= 0;
    return;
end
        
bestgain= -100;
model = struct;
% Go over all applicable classifiers and generate candidate weak models
for classf = classifierID

    modelCandidate= struct;    
    maxgain= -1;

    if classf == 1
        % Decision stump

        % proceed to pick optimal splitting value t, based on Information Gain  
        for q= 1:numSplits
            
            if mod(q-1,5)==0
                r= randi(D);
                col= X(:, r);
                tmin= min(col);
                tmax= max(col);
            end
            
            t= rand(1)*(tmax-tmin)+tmin;
            dec = col < t;
            Igain = evalDecision(Y, dec, u);

            if Igain>maxgain
                maxgain = Igain;
                modelCandidate.r= r;
                modelCandidate.t= t;
            end
        end

    elseif classf == 2
        % Linear classifier using 2 dimensions

        % Repeat some number of times: 
        % pick two dimensions, pick 3 random parameters, and see what happens
        for q= 1:numSplits

            r1= randi(D);
            r2= randi(D);
            w= randn(3, 1);
            
            dec = [X(:, [r1 r2]), ones(N, 1)]*w < 0;
            Igain = evalDecision(Y, dec, u);
            
            if Igain>maxgain
                maxgain = Igain;
                modelCandidate.r1= r1;
                modelCandidate.r2= r2;
                modelCandidate.w= w;
            end
        end

    elseif classf == 3
        % Conic section weak learner in 2D (not too good presently, what is the
        % best way to randomly suggest good parameters?

        % Pick random parameters and see what happens
        for q= 1:numSplits

            if mod(q-1,5)==0
                r1= randi(D);
                r2= randi(D);
                w= randn(6, 1);
                phi= [X(:, r1).*X(:, r2), X(:,r1).^2, X(:,r2).^2, X(:, r1), X(:, r2), ones(N, 1)];
                mv= phi*w;
            end
            
            t1= randn(1);
            t2= randn(1);
            if rand(1)<0.5, t1=-inf; end
            dec= mv<t2 & mv>t1;
            Igain = evalDecision(Y, dec, u);

            if Igain>maxgain
                maxgain = Igain;
                modelCandidate.r1= r1;
                modelCandidate.r2= r2;
                modelCandidate.w= w;
                modelCandidate.t1= t1;
                modelCandidate.t2= t2;
            end
        end

    elseif classf==4
        % RBF weak learner: Picks an example and bases decision on distance
        % threshold
        
        % Pick random parameters and see what happens
        for q= 1:numSplits

            % this is expensive, lets only recompute every once in a while...
            if mod(q-1,5)==0
                x= X(randi(size(X, 1)), :);
                dsts= pdist2(X, x);
                maxdsts= max(dsts);
                mindsts= min(dsts);
            end

            t= rand(1)*(maxdsts - mindsts)+ mindsts;
            dec= dsts < t;
            Igain = evalDecision(Y, dec, u);

            if Igain>maxgain
                maxgain = Igain;
                modelCandidate.x= x;
                modelCandidate.t= t;
            end
        end

    else
        fprintf('Error in weak train! Classifier with ID = %d does not exist.\n', classf);
    end

    % see if this particular classifier has the best information gain so
    % far, and if so, save it as the best choice for this node
    if maxgain >= bestgain
        bestgain = maxgain;
        model= modelCandidate;
        model.classifierID= classf;
    end

end

end

function Igain= evalDecision(Y, dec, u)
% gives Information Gain provided a boolean decision array for what goes
% left or right. u is unique vector of class labels at this node

    YL= Y(dec);
    YR= Y(~dec);
    H= classEntropy(Y, u);
    HL= classEntropy(YL, u);
    HR= classEntropy(YR, u);
    Igain= H - length(YL)/length(Y)*HL - length(YR)/length(Y)*HR;

end

% Helper function for class entropy used with Decision Stump
function H= classEntropy(y, u)

    cdist= histc(y, u) + 1;
    cdist= cdist/sum(cdist);
    cdist= cdist .* log(cdist);
    H= -sum(cdist);
    
end
