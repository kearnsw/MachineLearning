Markov Chain Monte Carlo
================
Implementation of the Metropolis-Hastings Algorithm in Python

Generate data
----------------
./makedata -h
./makedata --mu 2 --sigma 1 --samples 1000

Data is stored in data/data.out

Run MCMC
----------------
./mcmc -h
./mcmc --mu_min -10 --mu_max 10 --sigma_min 0.1 --sigma_max 10 --stepsize 0.01 --samples 10000 > data/10000.out

Plot 
----------------
./heatmap data/10000.out
./makeHistMu data/10000.out

My graphs are stored in the graphs directory

f) The MAP values are close to the parameters given to the makedata function. We only sampled 1000 data points from that distribution, so perhaps the estimate could reflect the true value of mu. As we increase the number of steps the credible interval shrinks and therefore the confidence in the prediction increases. 

Ex.
./findMAP data/86239.out 
mu                                      2.04402
sigma                                   1.05269
  
g) The mean values are thrown off not only by the initial estimates of mu and local maxima. 

Ex.
./findMean data/10000.out 
mu: 2.1968637966115097 
sigma: 1.1249106360591532

For 95% area under the curve: 
./findCredibleInterval -i data/86239.out -p 2.5

