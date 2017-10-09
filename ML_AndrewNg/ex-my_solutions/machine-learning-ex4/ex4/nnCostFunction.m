function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%printf("num labels is %d\n", num_labels);
K = num_labels;

%Theta1 %(25x401) %to help reckoning
%Theta2 %(10x 26) %only for this particular data set

a1=[ones(m,1), X];  %a1=(5000,401)
%printf("size a1 %dx%d\n", size(a1,1), size(a1,2));
for i=1:m
    a1i=a1(i,:);  %(1,401)
    z2i=Theta1*a1i'; %' (25,401 x 401,1) = (25,1)
    a2i=sigmoid(z2i); %(25,1)
    a2i_aug=[1;a2i];  %(26,1)
    z3i=Theta2*a2i_aug;  %(10,26 x 26,1) = (10,1)
    a3i=sigmoid(z3i); %(10,1)
    %printf("y(i) is %d\n",y(i));
    yk=recode(y(i)); %(10,1)
    var1 = (yk.*log(a3i));
    var2 = (1-yk).*(log(1-a3i));
    var3=-var1-var2; %(10,1)
    var4=sum(var3);
    J=J+var4;
endfor

J = J/m;

% Helper to convert a value to vector of [1,0,0,0,0,...0] form
function [yk] = recode(y)
    yk=zeros(K,1);
    yk(y)=1;
endfunction


%%
%% Regularized Cost Function J
%%
var5= Theta1(:,[2:size(Theta1,2)]);
var6=var5.^2;
var7=sum(var6(:)); %sum all elements

var8=Theta2(:,[2:size(Theta2,2)]);
var9=var8.^2;
var10=sum(var9(:));

var11 = ((var7+var10)*lambda)/(2*m);
J=J+var11;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%Theta1 %(25x401) %to help reckoning
%Theta2 %(10x 26) %only for this particular data set

for i=1:m
    a1i=a1(i,:);  %(1,401)
    z2i=Theta1*a1i'; %' (25,401 x 401,1) = (25,1)
    a2i=sigmoid(z2i); %(25,1)
    a2i_aug=[1;a2i];  %(26,1)
    z3i=Theta2*a2i_aug;  %(10,26 x 26,1) = (10,1)
    a3i=sigmoid(z3i); %(10,1)
    yk=recode(y(i)); %(10,1)
    
    d3i=a3i-yk; %(10,1);
    d2i=(Theta2'*d3i).*sigmoidGradient([1;z2i]); %' (26,1)
    Theta2_grad = Theta2_grad + d3i*a2i_aug'; %'
    d2i_sub=d2i(2:end,:);
    Theta1_grad = Theta1_grad + d2i_sub*a1i;
endfor

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_forreg = Theta2;
Theta2_forreg(:,1)=0;
Theta1_forreg = Theta1;
Theta1_forreg(:,1) = 0;
Theta2_grad = Theta2_grad + Theta2_forreg *(lambda/m);
Theta1_grad = Theta1_grad + Theta1_forreg *(lambda/m);




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
