library("bbob")



# par - a numeric vector with the starting point for the optimization
# fun - the function to be minimized
# lower, upper - the bounds of the box constraints
# max_eval - the number of function evaluations left in the allocated budget
optimizer.wrapper <- function(par, fun, lower, upper, max_eval) {
  optim(par, fun, method="L-BFGS-B",
        lower=lower, upper=upper,
        control=list(maxit=max_eval))
}

# 2nd - algorithm id (name)
# 3rd - the name of the base directory where the result files will be stored
bbo_benchmark(optimizer.wrapper, "mlp-approx-ev-opt", "optim_mlp-aprox-ev-opt", budget=10000)
