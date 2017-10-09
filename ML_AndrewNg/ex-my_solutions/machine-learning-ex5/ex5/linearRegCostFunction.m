function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

X;
y;
theta;
hX = X*theta;
error=hX-y;
error2 = error.^2;
var1 = sum(error2);
var2 = var1 / (2*m);

theta2 = theta.^2;
theta2(1,:)=0;
var3 = sum(theta2);

J = var2 + (lambda/(2*m))*var3;

var4 = X' * error;
var5 = var4/m;

grad0=grad(1,:); % copy over row for theta0
grad = grad + ((lambda/m)*theta);
grad(1,:)=grad0; %copy back theta0
grad = var5+grad;

% =========================================================================

end
