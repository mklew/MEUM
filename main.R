library("bbob")


## algorytm ewolucyjny
ev.optim <- function (lambda, mu, crossover.probability,
                      func, lower, uppper, dimension
                      init, stop.criteria,
                      selection, crossover, mutation, replacement) {
  p = init() # p - wektor wektorow (nie tablica/macierz!)
  while (TRUE) {
    for(i in 1:lambda) {
      if(runif(dimension) < crossover.probabilty) 
        o = mutation(crossover(selection(p, 2))) 
      else o = mutation(selection(p, 1))
    }
    p = replacement(p, o)
  }
}

ev.rand.init <- function(mu, par, eps) replicate(mu, rnorm(length(par), mean=par, sd=eps), simplify=FALSE)

# selekcja turniejowa
ev.t.selection <- function(p, number, func) {
  qn = 3 # liczba punktow losowanych do turnieju
  one.t = function() {
  # pojedynczy turniej
    q = sample(p, qn, replace=FALSE)
    print(q)
    results = sapply(q, func)
    print(results)
    q[which.max(results)]
  }
  replicate(number, one.t())
}



# par - wektor numeryczny z punktem startowym dla optymalizacji (moze byc zignorowane)
# fun - minimalizowana funkcja
# lower, upper - ograniczenia punktow z dziedziny (granice kostki)
# max_eval - pozostala liczba ewaluacji funkcji dla obecnego stanu budzetu
optimizer.wrapper <- function(par, fun, lower, upper, max_eval) {
  print("par")
  print(par) # wziac liczbe wymiarow z par!
  print("lower")
  print(lower)
  print("upper") 
  print(upper)
  print("max_eval")
  print(max_eval)
  #optim(par, fun, method="L-BFGS-B",
  #      lower=lower, upper=upper,
  #      control=list(maxit=max_eval))
}

# 2gi- id (nazwa) algorytmu
# 3ci - nazwa katalogu, do ktorego beda zapisane wyniki
bbo_benchmark(optimizer.wrapper, "mlp-model-opt", "optim_mlp-model-opt", budget=10000)
