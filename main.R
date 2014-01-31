library("bbob")
library('neuralnet')

## algorytm ewolucyjny
ev.optim <- function (par, lambda, mu, crossover.probability,
                      func, dimension,
                      init, stop.criteria,
                      selection, crossover, mutation, replacement) {
  t = 1
  p = init(mu, par) # p - wektor wektorow (nie tablica/macierz!)
  best = list()
  best = c(best, list(p[[which.min(sapply(p, func))]]))
  repeat {
    t = t+1
    o = list()
    for(i in 1:lambda) {
      if(runif(1) < crossover.probability) 
        oi = mutation(crossover(selection(p, 2, func))) 
      else oi = mutation(selection(p, 1, func))
      o = c(o, list(oi))
    }
    p = replacement(p, o, func=func, mu=mu)
    best = c(best, list(p[[which.min(sapply(p, func))]]))
    if(stop.criteria(t, best, func)) {
      break
    }
  }
  best[[t]]
}

ev.init.rand <- function(mu, par, eps) replicate(mu, rnorm(length(par), mean=par, sd=eps), simplify=FALSE)

# Kryterium minimalnej szybkosci poprawy
ev.stop.criteria <- function(t, best, func, tmax, tau, eps) {
  t >= tmax || (t-tau >= 1 && abs(func(best[[t-tau]]) - func(best[[t]])) < eps)
}

# selekcja turniejowa
ev.selection.t <- function(p, number, func) {
  qn = 5 # liczba punktow losowanych do turnieju
  one.t = function() {
  # pojedynczy turniej
    q = sample(p, qn, replace=FALSE)
    results = sapply(q, func)
    q[which.min(results)]
  }
  replicate(number, one.t())
}

# krzyzowanie jednopunktowe
ev.crossover.uni <- function(pair) {
  w = sample(c(0,1), length(pair[[1]]), replace=TRUE)
  list(w*pair[[1]] + (1-w)*pair[[2]])
}

# Prosta mutacja przy uzyciu ustalonego sigma
ev.mutation <- function(x, sigma) {
  point = x[[1]]
  point + rnorm(length(point), mean=0, sd=sigma)
}

# prev - inaczej Pt
# cur - inaczej Ot
# k - liczba najlepszych osobnikow z poprzedniej (prev) populacji, ktore dolacza do nowej populacji
# mu - rozmiar wynikowej populacji
ev.replacement.elite <- function(prev, cur, func, mu, k) {
  best.prev = prev[order(sapply(prev, func))][1:k] # nieoptymalne w przypadku, gdy k jest niewielkie (ale te rozwiazanie jest najbardziej ogolne)
  cand = c(best.prev, cur)
  cand[order(sapply(cand, func))][1:mu]
}

replmodel.ev.optim <- function(par, func, dimension) {
  ev.optim(par=par, func=func, dimension=dimension,
           lambda = globalParameters.lambda, # TODO na podstawie czego ustalic lambda i mu?
           mu = globalParameters.mu,
           crossover.probability = 0.5,
           init = function(mu, par) ev.init.rand(mu, par, eps=0.1),
           stop.criteria = function(t, best, func) ev.stop.criteria(t, best, func, tmax=globalParameters.tmax, tau=10, eps=0.1),
           selection = ev.selection.t,
           crossover = ev.crossover.uni,
           mutation = function(x) ev.mutation(x, 0.5),
           replacement = function(p, c, func, mu) ev.replacement.elite(p, c, func, mu, 1))
}

initial.point <- function(dimension, n, lower, upper) {
  # TODO czy te kostki to będą dla każdego x? tzn. jeśli mamy f(x1,x2) to czy kostki będa osobne dla x1 i x2
  t(sapply(1:n, function(i) runif(dimension, min=lower, max=upper)))
}

# names - wektor z nazwami data frame. Ostatnia nazwa symbolizuje Y. Reszta to features
nn.model.formula <- function(names) {
  yName <- names[length(names)]
  featureNames <- names[1:length(names)-1]
  features <- paste(featureNames, collapse="+")
  formulaStr <- paste(c(yName,"~",features), collapse='')  
  formula(formulaStr)
}

# zwraca funkcje aproksymującą dla pojedynczych wektorów
nn.model.approx <- function(nn) {
  approx <- function(arg) {
    nn.results <- compute(nn, arg)
    nn.results$net.result
  }
  wrapper <- function(vectorOrMatrix) {
    if(is.matrix(vectorOrMatrix)){
      approx(vectorOrMatrix)
    }
    else {
      m <- array(vectorOrMatrix, dim=c(1, length(vectorOrMatrix)))
      approx(m)
    }  
  }
  wrapper
}

nn.model <- function(dataFrame, startWeights = FALSE) {  
  hiddenUnits <- globalParameters.hiddenUnits
  if(is.list(startWeights)) {
    neuralnet(nn.model.formula(names(dataFrame)), dataFrame, startweights=startWeights,
      hidden=hiddenUnits, threshold=0.01, err.fct="sse",act.fct="tanh", linear.output=TRUE)
  }
  else {
    # Tutaj jest problem jak ćwiczmy sieć mając tylko 1 punkt. 
    # W krokach stepmax nie jesteśmy w stanie osiągnąć tego thresholdu
    neuralnet(nn.model.formula(names(dataFrame)), dataFrame, rep=1, stepmax = 1e+05, lifesign='full',
      hidden=hiddenUnits, threshold=0.01, err.fct="sse",act.fct="tanh", linear.output=TRUE)
  }  
}

# n - liczba przykladow do wyprodukowania
explore.f <- function(fun, fDimension, n, lower, upper) {
  initialPoints <- t(sapply(1:n, function(i) runif(fDimension, lower, upper)))
  y0 <- apply(initialPoints,1, fun)
  as.data.frame(cbind(initialPoints, y0))  
}

exploit.f <- function(fun, fDimension, n, lower, upper) {
  points <- t(sapply(1:n, function(i) rnorm(fDimension, c((upper-lower)/10, (upper-lower)/10)))) # TODO poszukac lepszego rozwiazania dla sd
  y0 <- apply(points,1, fun)
  as.data.frame(cbind(points, y0))  
}

nn.normalize <- function(x) {
  m <- colMeans(x)
  colSd <- function (x, na.rm=FALSE) apply(X=x, MARGIN=2, FUN=sd, na.rm=na.rm)
  s <- colSd(x)
  data <- do.call("rbind", by(x, 1:nrow(x), function(row) {
      (row - m) / s
  }))

  denorm <- function(x) {
    x * s[1:length(x)] + m[1:length(x)]
  }
  list("data"=data, "mean"=m, "sd"=s, "denorm"=denorm)
}

nn.has.no.weights <- function(nn) {
  is.na(nn$weights)
}

optimizer.mse <- function(dataPoints, approximationFunction) {
  approxY <- approximationFunction(dataPoints[,1:length(dataPoints)-1])
  sqrt(mean((approxY - dataPoints[,length(dataPoints)])^2))
}

# parametry zmienna globalna
resetToDefaults <- function() {
  globalParameters.hiddenUnits <- 10
  globalParameters.nEval <- 50
  globalParameters.maxNWorseIters <- 3 # przez ile iteracji mozemy nie otrzymac poprawy
  globalParameters.nWorseIters <- 0 # obecna liczba iteracji bez poprawy aproksymacji funkcji celu
  globalParameters.maxIters <- 10 # maksymalna liczba iteracji 
  globalParameters.mu <- 100
  globalParameters.lambda <- 100
  globalParameters.tmax <- 10000 # maksymalna liczba populacji algorytmu ewolucyjnego
}

# par - wektor numeryczny z punktem startowym dla optymalizacji (moze byc zignorowane)
# fun - minimalizowana funkcja
# lower, upper - ograniczenia punktow z dziedziny (granice kostki)
# max_eval - pozostala liczba ewaluacji funkcji dla obecnego stanu budzetu
optimizer.wrapper <- function(par, fun, lower, upper, max_eval) {
  nEval <- globalParameters.nEval # liczba ewaluacji funkcji celu
  maxNWorseIters <- globalParameters.maxNWorseIters # przez ile iteracji mozemy nie otrzymac poprawy
  nWorseIters <- globalParameters.nWorseIters # obecna liczba iteracji bez poprawy aproksymacji funkcji celu
  maxIters <- globalParameters.maxIters # maksymalna liczba iteracji 
  
  fDimension <- length(par)
    # data frame dla sieci neuronowej
  dataPoints <- explore.f(fun, fDimension, nEval, lower, upper)  
  normalized <- nn.normalize(dataPoints)
  nnTrainingData <- normalized$data
  nn <- nn.model(nnTrainingData)
  approximationFunction <- nn.model.approx(nn)
  lastMse <- optimizer.mse(dataPoints,approximationFunction)
  bestPoint <- par
  t <- 0
  for(i in 1:maxIters) {
    print(paste("iteration ", i))  
    best <- replmodel.ev.optim(par, approximationFunction, fDimension)
    denormalized <- normalized$denorm(best)
    bestPoint <- denormalized 
    y <- fun(denormalized) # i+1 ewaluacja (+1 bo y0) 
    dataPoints <- rbind(dataPoints, c(denormalized,y))
    dataPoints <- rbind(dataPoints, exploit.f(fun, fDimension, nEval, lower, upper))
    normalized <- nn.normalize(dataPoints)
    nnTrainingData <- normalized$data    
    nn <- nn.model(nnTrainingData, nn$weights) # dotrenowujemy sieć wagi poczatkowe już mamy
    if(is.na(nn) || is.null(nn$weights)) {
      print("siec neuronowa nie posiada wag")
      break;
    }
    approximationFunction <- nn.model.approx(nn)
    
    currentMse <- optimizer.mse(dataPoints,approximationFunction)
    if(currentMse > lastMse) {
      if(nWorseIters > maxNWorseIters) {
        print("Osiagnieto maksymalna liczbe gorszych wartosci MSE")
        break
      } else {
        nWorseIters <- nWorseIters+1
      }
    } else nWorseIters <- 0 
    lastMse <- currentMse
  }
  ## TODO return denormalized vector
  bestPoint
}

ev.wrapper <- function(par, fun, lower, upper, max_eval) {
    wynik = replmodel.ev.optim(par, fun, length(par))
    print(wynik)
    wynik
}


ff <- function(x) {
  f <- function(mat) {
    rowSums(mat)^3 + 100  
  }
  if(is.matrix(x)){
    f(x)
  }
  else {
    m <- array(x, dim=c(1, length(x)))
    f(m)
  }  
}

#print(optimizer.wrapper(c(0,0), ff, -100, 100, 1000))

globalParameters.hiddenUnits <- 10
globalParameters.nEval <- 50
globalParameters.maxNWorseIters <- 3 # przez ile iteracji mozemy nie otrzymac poprawy
globalParameters.nWorseIters <- 0 # obecna liczba iteracji bez poprawy aproksymacji funkcji celu
globalParameters.maxIters <- 10 # maksymalna liczba iteracji 
globalParameters.mu <- 100
globalParameters.lambda <- 100
globalParameters.tmax <- 10000
resetToDefaults()
#bbo_benchmark(optimizer.wrapper, "mlp-model-opt", "optim_mlp-model-opt_15_16", budget=100, instances=c(15, 16), dimensions=c(5,10,20))

globalParameters.hiddenUnits <- 50
# globalParameters.nEval <- 50
# globalParameters.maxNWorseIters <- 3 # przez ile iteracji mozemy nie otrzymac poprawy
# globalParameters.nWorseIters <- 0 # obecna liczba iteracji bez poprawy aproksymacji funkcji celu
# globalParameters.maxIters <- 10 # maksymalna liczba iteracji 
# globalParameters.lambda <- 5
# globalParameters.mu <- 5
# globalParameters.tmax <- 1000 # liczba ewaluacji alg. ew. dopóki się nie podda
#bbo_benchmark(optimizer.wrapper, "mlp-model-opt", "optim_mlp-model-opt_15_16_units50", budget=100, instances=c(15, 16), dimensions=c(5,10,20))

# globalParameters.nEval <- 50
# globalParameters.maxNWorseIters <- 3 # przez ile iteracji mozemy nie otrzymac poprawy
# globalParameters.nWorseIters <- 0 # obecna liczba iteracji bez poprawy aproksymacji funkcji celu
# globalParameters.maxIters <- 10 # maksymalna liczba iteracji 
globalParameters.lambda <-250
globalParameters.mu <- 250
# globalParameters.tmax <- 1000 # liczba ewaluacji alg. ew. dopóki się nie podda
#bbo_benchmark(optimizer.wrapper, "mlp-model-opt", "optim_mlp-model-opt_15_16_lambda_25_mu_20", budget=100, instances=c(15, 16), dimensions=c(5,10,20))
bbo_benchmark(ev.wrapper, "mlp-model-opt", "optim_mlp-model-opt_ev_25_20", budget=100, dimensions=c(5,10,20))


