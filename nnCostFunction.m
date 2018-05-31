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
X = [ones(size(X,1),1) X];
         
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
a1 = sigmoid(Theta1*X');
a1 = [ones(1, size(a1,2)); a1];
a2 = sigmoid(Theta2*a1);

K = zeros(num_labels, 1);
for i=1:num_labels
    K(i) = i;
end;
Y = zeros(num_labels, m);

for i=1:m
    Y(:,i) = logical(K==y(i));
end;

res = Y.*log(a2) + (1-Y).*log(1-a2);
res = sum(res);
T1 = Theta1(:,2:end) .^ 2;
T2 = Theta2(:,2:end) .^ 2;

res = sum(res);
J = res*(-1/m) + (lambda/(2*m))*(sum(sum(T1,2))+sum(sum(T2,2)));

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
del1 = zeros(hidden_layer_size, input_layer_size+1);
del2 = zeros(num_labels, hidden_layer_size+1);
T = X';
for i=1:m
    delta3 = a2(:,i) - Y(:,i);
    Z2 = [1;Theta1*T(:,i)];
    delta2 = (Theta2'*delta3) .* sigmoidGradient(Z2);
    delta2 = delta2(2:end);
    del1 = del1 + delta2*(T(:,i)');
    del2 = del2 + delta3*(a1(:,i)');
end

Theta1_grad(:,2:end) = (del1(:,2:end) + lambda*Theta1(:,2:end))/m;
Theta1_grad(:,1) = del1(:,1)/m;

Theta2_grad(:,2:end) = (del2(:,2:end) + lambda*Theta2(:,2:end))/m;
Theta2_grad(:,1) = del2(:,1)/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

grad = [Theta1_grad(:); Theta2_grad(:)];


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

