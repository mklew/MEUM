# Gotowa sieć neuronowa z pakietu neuralnet

library('neuralnet')

traininginput <- as.data.frame(runif(50, min=0, max=100))
trainingoutput <- sqrt(traininginput)

trainingdata <- cbind(traininginput, trainingoutput)
colnames(trainingdata) <- c("Input", "Output")

# response variables ~ sum of covariates, inaczej mówiąc Y~x1+x2+x3+x3
nn <- neuralnet(Output~Input, trainingdata, hidden=10, threshold=0.01, err.fct="sse",act.fct="tanh", linear.output=T)

plot(nn)

testdata <- as.data.frame((1:10)^2)
results <- compute(nn, testdata)

ls(results)

print(results$net.result)

cleanoutput <- cbind(testdata, sqrt(testdata), as.data.frame(results$net.result))
colnames(cleanoutput) <- c("Input", "Expected Output", "Neural Net Output")

print(cleanoutput)
