
Andrej's Machine Learning toolbox 
by Andrej Karpathy (@karpathy)

---------------------------------------------------------------------------
Usage:

Random Forests for classification: (see demo for more)
opts.classfierID= [2, 3]; % use both 2D-linear weak learners (2) and conic (3)
m= forestTrain(X, Y, opts);
yhat = forestTest(m, X);
fprintf('Training accuracy = %.2f\n', mean(yhat==Y));

SVMs for classification: (see demo for more)
opts.C= 1e-2;
opts.addOnes= true;
opts.type= 4; % use L2-svm (squared hinge loss)
m= svmTrain(X, Y, opts);
yhat= svmTest(m, X);

---------------------------------------------------------------------------
More info:

Currently contains random forests and SVM's for classification. 
The Random Forest code is not industrial strength implementation and is currently meant more for research purposes. However, it is possible I'll want to speed this up in the near future and make it more robust.

Inspired by MSR's recent work on Random Forests:
https://research.microsoft.com/apps/pubs/default.aspx?id=155552

See http://cs.stanford.edu/~karpathy/randomForestSpiral.png
for results on spiral using 2-D linear weak learners. (Code that generates
the image is in forestdemo.m) 

---------------------------------------------------------------------------
Adding your own weak learners in Ranfom Forests:

It is fairly easy to add your own weak learners. Modify:
weakTrain.m:  add another elseif statement for classf variable, and put in
              code for your weak learner. Store all variables you need during
              test time in modelCandidate
weakTest.m:   add another elseif for your classifier, and implement the decision
              procedure, using variables you stored inside model.
Now just include your new classifier when setting opts.classfierID!
---------------------------------------------------------------------------

BSD Licence. 

