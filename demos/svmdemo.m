% Requires: minFunc
% http://www.di.ens.fr/~mschmidt/Software/minFunc/minFunc.html
% direct link: http://www.di.ens.fr/~mschmidt/Software/minFunc/minFuncExamples.zip

%% generate data

rand('state', 0);
randn('state', 0);
N= 50;
D= 2;

X1 = mgd(N, D, [4 3], [2 -1;-1 2]);
X2 = mgd(N, D, [1 1], [2 1;1 1]);
X3 = mgd(N, D, [3 -3], [1 0;0 4]);

X= [X1; X2; X3];
X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];

scatter(X(:,1), X(:,2), 20, Y)

%% svm 3-way classify

rand('state', 0);
randn('state', 0);

opts= struct;
opts.C= 1e-2;
opts.polyOrder= 2;
opts.rbfScale= 1;
for t=1:4
    
    opts.type= t; %1= linear, 2= poly, 3= rbf, 4= L2-svm
    if(t==1 || t==4), opts.addOnes= true;
    else opts.addOnes= false; 
    end
        
    tic;
    m= svmTrain(X, Y, opts); % train
    timetrain= toc;
    tic;
    yhatTrain = svmTest(m, X); % test
    timetest= toc;

    % plot results
    subplot(2,2,t);
    
    xrange = [-1.5 1.5];
    yrange = [-1.5 1.5];
    inc = 0.02;
    [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
    image_size = size(x);
    xy = [x(:) y(:)];

    yhat = svmTest(m, xy);
    decmaphard= reshape(yhat, image_size);

    imagesc(xrange,yrange,decmaphard);
    hold on;
    set(gca,'ydir','normal');
    cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
    colormap(cmap);
    plot(X(Y==1,1), X(Y==1,2), 'o', 'MarkerFaceColor', [.9 .3 .3], 'MarkerEdgeColor','k');
    plot(X(Y==2,1), X(Y==2,2), 'o', 'MarkerFaceColor', [.3 .9 .3], 'MarkerEdgeColor','k');
    plot(X(Y==3,1), X(Y==3,2), 'o', 'MarkerFaceColor', [.3 .3 .9], 'MarkerEdgeColor','k');
    hold off;
    fprintf('Type %d, Train time: %.2fs, Test time: %.2fs\n', t, timetrain, timetest);

    title(sprintf('Type= %d, Train accuracy: %f\n', t, mean(yhatTrain==Y)));
end