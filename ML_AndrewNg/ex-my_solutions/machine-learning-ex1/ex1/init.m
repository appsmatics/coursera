%% =================== Part 3: Gradient descent ===================
clear ; close all; clc
fprintf('Initializing data set\n')

data = load('ex1data1.txt');

y = data(:, 2);
m = length(y); % number of training examples

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% compute and display initial cost
computeCost(X, y, theta)


