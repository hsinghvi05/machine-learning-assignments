function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

thet=[0;theta(2:size(theta))];

h=sigmoid(X*theta);
p=lambda*(thet'*thet)/2/m;
%sdfg'
J=-1/m*(sum(y.*log(h) +(1-y).*log(1-h)))+p;

%grad(1)=1/m*(sum(h-y));
%grad(2)=1/m*(sum((h-y).*X(:,2))+lambda*theta(2));
%grad(3)=1/m*(sum((h-y).*X(:,3))+lambda*theta(3));
%grad(4)=1/m*(sum((h-y).*X(:,4))+lambda*theta(4));
%grad(5)=1/m*(sum((h-y).*X(:,5))+lambda*theta(5));

grad = (X'*(h - y)+lambda*thet)/m;



% =============================================================

end
