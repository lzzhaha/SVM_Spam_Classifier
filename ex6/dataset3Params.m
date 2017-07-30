function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sig_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%

err = 0;
for i=1:size(C_vec) 
  for j=1:size(sig_vec) 
    model = svmTrain(X,y,C_vec(i),@(x1, x2) gaussianKernel(x1, x2, sig_vec(j)));
    predictions = svmPredict(model,Xval);
    cur_err = mean(double(predictions~=yval));
    if((i==1 && j==1) || (err>cur_err))
      err = cur_err;
      C = C_vec(i);
      sigma = sig_vec(j);
    endif
  endfor
endfor
  





% =========================================================================

end
