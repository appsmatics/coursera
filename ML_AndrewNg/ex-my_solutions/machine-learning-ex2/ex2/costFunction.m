function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta

var1 = sigmoid(X*theta); % m x 1
var2 = log(1-var1);
var3 = (1-y).*var2;
var4 = log(var1);
var5 = y .* var4;
var6 = -var5 -var3;
J=sum(var6)/m;

var7 = (var1-y);
var8=(var7'*X)/m; % (n+1) x 1
grad=var8';

% =============================================================

endfunction
