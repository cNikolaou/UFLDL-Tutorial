function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

[netIn, trExamples] =  size(data);

m = trExamples;

partCost = 0;

DW1 = zeros(size(W1));
DW2 = zeros(size(W2));
Db1 = zeros(size(b1));
Db2 = zeros(size(b2));

% compute rho
y = data;
x = y;
a1 = x;
z2 = W1*a1 + repmat(b1,1,size(a1,2));
a2 = sigmoid(z2);

rho = 1/size(x,2)*sum(a2,2);

% Forward propagation
y = data;
x = y;
a1 = x;
z2 = W1*a1 + repmat(b1,1,m);
a2 = sigmoid(z2);
z3 = W2*a2 + repmat(b2,1,m);
a3 = sigmoid(z3);
hx = a3;

% Back propagation
partCost = 1/2*sqrt(sum(abs(hx-y).^2,1)).^2;%sum(1/2*sqrt(dot(hx-y,hx-y,2)).^2);%trace(1/2*sqrt((hx - y)'*(hx - y)).^2);

delta3 = -(y-a3).*a3.*(1-a3);
delta2 = (W2'*delta3 +...
    repmat(beta*(-sparsityParam./rho + (1-sparsityParam)./(1-rho)),1,m)).*a2.*(1-a2);

DW1 = delta2*a1';
Db1 = sum(delta2,2);
DW2 = delta3*a2';
Db2 = sum(delta3,2);

% Compute cost:
% with lambda = 0 and beta = 0
% cost = 1/m * partCost;
% with beta = 0;
cost = 1/m * sum(partCost) + lambda/2 * sum([sum(sum(W1.^2)) sum(sum(W2.^2))]);

% Add sparsity term
cost = cost + beta*sum(compKL(sparsityParam,rho));

% Compute W1grad:
% with lambda = 0 and beta = 0
% W1grad = 1/m * DW1;
% with beta = 0;
W1grad = 1/m * DW1 + lambda * W1;

% Compute W2grad:
% with lambda = 0 and beta = 0
% W2grad = 1/m * DW2;
% with beta = 0;
W2grad = 1/m * DW2 + lambda * W2;

% Compute b1grad:
% with lambda = 0 and beta = 0
% b1grad = 1/m * Db1;
% with beta = 0;
b1grad = 1/m * Db1;

% Compute b2grad:
% with lambda = 0 and beta = 0
% b2grad = 1/m * Db2;
% with beta = 0;
b2grad = 1/m * Db2;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function kl = compKL(sparsityParam, rho)

    kl = sum(sparsityParam.*log(sparsityParam./rho) +...
        (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

