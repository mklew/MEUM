library("bbob")


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
# TODO func nie powinno byc wolane tak czesto (wczesniej juz byly policzone wartosci funkcji dla tych punktow) - przechowywac obliczona wartosc w liscie reprezentujacej dany punkt!

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
# TODO tu mozna by zrobic z adaptacja kierunku spadku gradientu funkcji celu
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
           lambda = 5, # TODO na podstawie czego ustalic lambda i mu?
           mu = 5,
           crossover.probability = 0.5,
           init = function(mu, par) ev.init.rand(mu, par, eps=0.1),
           stop.criteria = function(t, best, func) ev.stop.criteria(t, best, func, tmax=1000, tau=10, eps=0.1),
           selection = ev.selection.t,
           crossover = ev.crossover.uni,
           mutation = function(x) ev.mutation(x, 0.5),
           replacement = function(p, c, func, mu) ev.replacement.elite(p, c, func, mu, 1))
}

replmodel.optim <- function() {
  
}

# par - wektor numeryczny z punktem startowym dla optymalizacji (moze byc zignorowane)
# fun - minimalizowana funkcja
# lower, upper - ograniczenia punktow z dziedziny (granice kostki)
# max_eval - pozostala liczba ewaluacji funkcji dla obecnego stanu budzetu
optimizer.wrapper <- function(par, fun, lower, upper, max_eval) {
  replmodel.ev.optim(par, fun, length(par))
}

# 2gi- id (nazwa) algorytmu
# 3ci - nazwa katalogu, do ktorego beda zapisane wyniki
bbo_benchmark(optimizer.wrapper, "mlp-model-opt", "optim_mlp-model-opt", budget=10000, instances=c(1))
