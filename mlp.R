# Perceptron wielowarstwowy

# 3 warstwy w tym 1 warstwa ukrywa, funkcja aktywacji tanh

gz <- function(z){ tanh(z) }

# pochodna po tanh
d.gz <- function(z){ 1 - (tanh(z))^2 }

# zwraca macierz o rozmiarze l.out x l.in + 1 z małymi losowymi wagami początkowymi
random.weights <- function(l.in, l.out) {
	epsilon <- 0.12 
	matrix(rnorm((l.in + 1) * l.out), nrow = l.out, ncol = l.in + 1)  * 2 * epsilon - epsilon
}

ones <- function(rows, cols) { matrix(replicate(m,cols), nrow=rows, ncol=cols) }
zeroes <- function(dim) { matrix(0, nrow = dim[1], ncol = dim[2]) }

addBias <- function(mtx) {
	m <- dim(mtx)[1]
	cbind(ones(m,1), mtx)
}

# dla 1 warstwy ukrytej, X nie zawiera jednostek bias
forward.propagation <- function(theta1, theta2, X) {		
	m <- dim(X)[1]
	X.with.bias <- cbind(ones(m, 1), X)
	a1 <- X.with.bias
	z2 <- a1 %*% t(theta1)
	a2 <- addBias(gz(z2))
	z3 <- a2 %*% t(theta2)
	a3 <- z3
	a3
}

errors <- function(h, y) {
	(h - y)^2
}

nnCostFunction <- function(thetaVec, input.layer.size, hidden.layer.size, X, y, lambda) {
	theta1Limit <- (input.layer.size + 1) * hidden.layer.size
	theta1 <- matrix(thetaVec[1: theta1Limit], hidden.layer.size, input.layer.size + 1)
	theta2 <- matrix(thetaVec[(theta1Limit + 1): length(thetaVec)], 1, hidden.layer.size + 1)

	m <- dim(X)[1]
	J <- 0
	X <- addBias(X)
	theta1.grad <- zeroes(dim(theta1))
	theta2.grad <- zeroes(dim(theta2))

	# forward propagation
	a1 <- X
	z2 <- a1 %*% t(theta1)
	a2 <- addBias(gz(z2))
	z3 <- a2 %*% t(theta2)
	a3 <- z3

	th1 <- theta1
	th1[,1] = 0
	th2 <- theta2
	th2[,1] = 0

	J <- (-1/m) * sum(errors(a3, y)) + (lambda / (2 * m)) * (sum(th1 ^ 2) + sum(th2 ^ 2))

	# back propagation
	d3 <- a3 - y
	d2 <- (d3 %*% theta2) * d.gz(addBias(z2))
	d2 <- d2[,2:dim(d2)[2]]

	theta1.grad <- 1/m * t(d2) %*% a1 + lambda * th1
	theta2.grad <- 1/m * t(d3) %*% a2 + lambda * th2

	c(as.vector(theta1.grad), as.vector(theta2.grad))
}

# sprawdzenie, że coś działa
number.of.examples <- 50
number.of.features <- 3
X <- matrix(1:(number.of.examples * number.of.features), nrow=number.of.examples,ncol=number.of.features)
y <- ones(dim(X)[1], 1)

inputLayerSize <- number.of.features
hiddenLayerSize <- 6
outputLayerSize <- 1

theta1 <- random.weights(inputLayerSize, hiddenLayerSize) # 6 x 4
theta2 <- random.weights(hiddenLayerSize, outputLayerSize) # 1 x 7

forward.propagation(theta1, theta2, X)

nnCostFunction(c(as.vector(theta1), as.vector(theta2)), inputLayerSize, hiddenLayerSize, X, y, 1)

