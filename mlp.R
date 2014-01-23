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

# dla 1 warstwy ukrytej, X nie zawiera jednostek bias
forward.propagation <- function(theta1, theta2, X) {	
	ones <- function(rows, cols) { matrix(replicate(m,cols), nrow=rows, ncol=cols) }
	m <- dim(X)[1]
	X.with.bias <- cbind(ones(m, 1), X)
	a1 <- X.with.bias
	z2 <- a1 %*% t(theta1)
	a2 <- cbind(ones(m,1), gz(z2))
	z3 <- a2 %*% t(theta2)
	a3 <- z3
	a3
}

# sprawdzenie, że coś działa
number.of.examples <- 50
number.of.features <- 3
X <- matrix(1:(number.of.examples * number.of.features), nrow=number.of.examples,ncol=number.of.features)

inputLayerSize <- number.of.features
hiddenLayerSize <- 6
outputLayerSize <- 1

theta1 <- random.weights(inputLayerSize, hiddenLayerSize)
theta2 <- random.weights(hiddenLayerSize, outputLayerSize)

forward.propagation(theta1, theta2, X)

