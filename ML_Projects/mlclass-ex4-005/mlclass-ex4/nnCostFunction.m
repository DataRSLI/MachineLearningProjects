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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%%%%%%%%Foward Propagation%%%%%%%%%%%%%%%%%%

% Apply Foward Propagation in order to compute 
% what out Hypothesis outputs given the input 
% x of our training set

% This is the Vectorized implemenation of forward 
% propogation and allows us to caluculate all of 
% the activation values for all of the the neurons 
% in the neural network

DeltaL1 = 0; % Will be used to compute par. derv. of J(Theta) 
DeltaL2 = 0; % wrt theta they will be used as accumulators
             % that will slowly add things through to compute
             % the pr. dev. of J(Theta)

for i=1:m,   % Assuming a large training set when implementing
             % back propagation this is a looping through the 
             % training set
           
% Activation_values_of_first_layer
 a_1 = [1 X(i,:)];
 z_2 = a_1 * Theta1';
% Activation_values_of_first_Hidden_Layer
 a_2 = [1 sigmoid(z_2)];
 z_3 = a_2 * Theta2';
% Activation_values_for output_of_the_Hypothosis
 a_3 = sigmoid(z_3);
 
%%%%%%%%%%%%%%%%%%%Cost Function Equation%%%%%%%%%%%%
 yk = zeros(num_labels,1);
 yk(y(i)) = 1;

 J = J + (-yk' * log(a_3')-(1 - yk')* log(1 - a_3'));
 %%%%%%%%%%%%%%%%%%%Cost Function Equation Reg%%%%%%%%%
 J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + ....
    sum(sum(Theta2(:,2:end).^2))); 

%%%%%%%%%%%%%%%%%%%Back Propagation%%%%%%%%%%%%%%%%%%

% Apply Back Propagation Algorithm in order to compute 
% derivatives which calculates the error of each node j 
% layer l for the delta representing this error for the 
% appropreiate node


% Vectorized implementation of Back Propagation:If you 
% think of delta a and y as a vector then you can take the
% del(lj) = a(lj) - y(lj)-> del(l) = a(l) - y who dimensions
% are equal to the number of the networks output units
 
% Error term for the last layer of the neural network
  delta_3 = a_3' - yk;

% Error term for the last layer - 1 of the neural network
  delta_2 = Theta2' * delta_3 .* sigmoidGradient([1 z_2])';

% There is no delta1 that is the input layer and has nor 
% error associated with it
  DeltaL1 = DeltaL1 + delta_2(2:end) * a_1; 
  DeltaL2 = DeltaL2 + delta_3 * a_2; 

end
J = J / m;

Theta1_grad = 1 / m * DeltaL1;
Theta2_grad = 1 / m * DeltaL2;

nonBiasedTheta1 = Theta1;
nonBiasedTheta1(:,1) = 0;
Theta1_grad = Theta1_grad + lambda / m * nonBiasedTheta1;

nonBiasedTheta2 = Theta2;
nonBiasedTheta2(:,1) = 0;
Theta2_grad = Theta2_grad + lambda / m * nonBiasedTheta2;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
