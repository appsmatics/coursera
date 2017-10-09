%% =================== Part 3: Gradient descent ===================
clear ; close all; clc
fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Some gradient descent settings
iterations = 400;
alpha = 0.01;
theta = zeros(3, 1);


