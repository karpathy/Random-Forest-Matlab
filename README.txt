
Andrej's Machine Learning toolbox 
by Andrej Karpathy (@karpathy)
---------------------------------------------------------------------------

Currently contains random forest for classification. This is not industrial
strength implementation and is currently meant more for research purposes and
fun. However, it is possible I'll want to speed this up in the near future 
and make it more robust.

Inspired by MSR's recent work on Random Forests:
https://research.microsoft.com/apps/pubs/default.aspx?id=155552

See http://cs.stanford.edu/~karpathy/randomForestSpiral.png
for what current spiral with this looks like, using 2-D linear weak learners.

------- Adding your own weak learners -------------------------------------
It is fairly easy to add your own weak learners. Modify:
weakTrain.m:  add another elseif statement for classf variable, and put in
              code for your weak learner. Store all variables you need during
              test time in modelCandidate
weakTest.m:   add another elseif for your classifier, and implement the decision
              procedure, using variables you stored inside model.
Now just include your new classifier when setting opts.classfierID!
---------------------------------------------------------------------------

BSD Licence. 

