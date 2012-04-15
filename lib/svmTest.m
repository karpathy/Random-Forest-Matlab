function Yhard = svmTest(model, X, opts)
    % used in conjunction with svmTrain
    
    if (model.addOnes), X= [X, ones(size(X,1), 1)]; end
    
    if model.type == 1
        [~, Yhard] = max(X*model.w,[],2);
        
    elseif model.type == 2
        K= kernelPoly(X, model.X, model.polyOrder);
        [~, Yhard] = max(K*model.uPoly,[],2);
        
    elseif model.type == 3
        K= kernelRBF(X, model.X, model.rbfScale);
        [~, Yhard] = max(K*model.urbf,[],2);
        
    elseif model.type == 4
        [~, Yhard] = max(X*model.w,[],2);
        
    else
        fprintf('Unrecognized type %d, exitting...\n', model.type);
        Yhard= 0;
        return;
    end
    