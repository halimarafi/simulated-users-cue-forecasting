// Forecast C, rung 0: how much does the cue shift the intercept, on average?
// Each trial's basic-trial traits are fixed (estimated in notebook 02); only the shift is estimated.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  vector[N] x_amt;                 // log(later amount / 20)
  vector[N] x_del;                 // log(delay)
  matrix[N, 3] th;                 // fixed traits per trial: intercept, amount slope, delay slope
}
parameters {
  real<lower=-5, upper=5> d_intercept;  // limit far outside any realistic cue effect
}
model {
  d_intercept ~ normal(0, 2);
  choice ~ bernoulli_logit(th[, 1] + d_intercept + th[, 2] .* x_amt + th[, 3] .* x_del);
}
