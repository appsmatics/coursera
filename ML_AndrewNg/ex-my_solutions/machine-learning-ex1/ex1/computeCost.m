function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

predictions = X*theta; % h(x)
err=(predictions-y);  % h(x)-y
squareError = err .^2;

sumOfSquareError = sum(squareError);

J= sumOfSquareError /(2*m); % (sigma(h(x)-y)^2 ) / (2*m)   

% =========================================================================

end
