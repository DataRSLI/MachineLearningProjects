function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %part_one = ((theta*X(1,:)'+ theta*X(2,:)')- y)*
%     Error = X * theta - y;
%     for i = 1:2
%         S(i) = sum(Error.*X(:,i));
%     end
% 
%     theta = theta - alpha * (1/m) * S';
    
    %================
%     temp0 = theta(1,1) - (alpha/m)*sum((X*theta-y).*X(:,1));
%     temp1 = theta(2,1) - (alpha/m)*sum((X*theta-y).*X(:,2));
%     
%     theta(1,1) = temp0;
%     theta(2,1) = temp1;
    %======================
    theta = theta - (alpha/m) * (X' * (X*theta-y));
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
