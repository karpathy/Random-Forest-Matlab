function model = svmTrain(X, Y, opts)
    % trains an SVM using minFunc
    % X is NxD features
    % Y is discrete array of labels that must (for now) be 1...K
    
    % opts.type specifies which method to use. 
    % 1= linear SVM
    % 2= polynomial Kernel SVM of order opts.polyOrder
    % 3= rbf Kernel SVM with scale parameter opts.rbfScale
    %    the scale is used in equation Z = 1/sqrt(2*pi*sigma^2);
    %    to scale the output of the exponential, where sigma is the scale
    % 4= l2svm, which unlike normal svm uses squared hinge loss
    
    % model is to be plugged in to yhat = svmTest(model, X)
    
    % TODO: Support labels that are not in 1..K
    % TODO: n-fold Cross validation support
    
    addOnes= false;
    C= 1;
    type= 1;
    rbfScale= 1;
    polyOrder= 2;
    if nargin < 3, opts= struct; end
    if isfield(opts, 'addOnes'), addOnes= opts.addOnes; end
    if isfield(opts, 'C'), C= opts.C; end
    if isfield(opts, 'type'), type= opts.type; end
    if isfield(opts, 'rbfScale'), rbfScale= opts.rbfScale; end
    if isfield(opts, 'polyOrder'), polyOrder= opts.polyOrder; end
    
    if(addOnes), X= [X, ones(size(X,1), 1)]; end
    
    [N, D]= size(X);
    u= unique(Y);
    K = length(u);
    
    minFuncOpts= struct;
    minFuncOpts.display= 0;
    
    if type == 1
    
        % Linear SVM
        funObj = @(w) SSVMMultiLoss(w, X, Y, K);
        wLinear = minFunc(@penalizedL2, zeros(D*K,1), minFuncOpts, funObj, C);
        model.w = reshape(wLinear,[D K]);

    elseif type == 2
    
        % Polynomial SVM
        Kpoly = kernelPoly(X, X, polyOrder);
        funObj = @(v) SSVMMultiLoss(v, Kpoly, Y, K);
        uPoly = minFunc(@penalizedKernelL2_matrix, randn(N*K,1), minFuncOpts, Kpoly, K, funObj, C);
        model.X = X; % must save all the training data... (ok only support vectors, todo)
        model.uPoly= reshape(uPoly, [N, K]);
        model.polyOrder= polyOrder;
        
    elseif type == 3
    
        % RBF SVM
        Krbf = kernelRBF(X, X, rbfScale);
        funObj = @(v) SSVMMultiLoss(v, Krbf, Y, K);
        uRBF = minFunc(@penalizedKernelL2_matrix, randn(N*K,1), minFuncOpts, Krbf, K, funObj, C);
        model.X = X; % must save all the training data... (ok only support vectors, todo)
        model.urbf= reshape(uRBF,[N, K]);
        model.rbfScale= rbfScale;
    
    elseif type == 4
        
        % L2 SVM (squares the slack variables, i.e. squared hinge loss)
        funObj = @(v) l2svmloss(v, X, Y, K, C);
        wLinear = minFunc(@penalizedL2, zeros(D*K,1), minFuncOpts, funObj, C);
        model.w = reshape(wLinear,[D K]);
        
    else
        
        fprintf('Unrecognized type %d, exitting...\n', type);
        model= struct;
        return; 
        
    end
    
    model.type= type;
    model.addOnes= addOnes;
    model.C= C;
    model.u= u;
    
    % 1-vs-all L2-svm loss function;  similar to LibLinear.
    % Originally taken from Adam Coates' code, slightly adapted
    function [loss, g] = l2svmloss(w, X, y, K, C)
        
      [M,N] = size(X);
      theta = reshape(w, N,K);
      Y = bsxfun(@(y,ypos) 2*(y==ypos)-1, y, 1:K);
      margin = max(0, 1 - Y .* (X*theta));
      loss = (0.5 * sum(theta.^2)) + C*sum(margin.^2);
      loss = sum(loss);  
      g = theta - 2 * C * (X' * (margin .* Y));
      g = g(:);

