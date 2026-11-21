// Forecast C, rung 1: how much does the cue shift patience, log(k), on average?
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  vector<lower=0>[N] amt;
  vector<lower=0>[N] delay;
  real<lower=0> ss_amt;
  matrix[N, 2] th;                 // fixed traits per trial: log k, log consistency
}
parameters {
  real<lower=-2, upper=2> d_logk;       // limit far outside any realistic cue effect
}
model {
  d_logk ~ normal(0, 1);
  vector[N] sv = amt ./ (1 + exp(th[, 1] + d_logk) .* delay);
  choice ~ bernoulli_logit(exp(th[, 2]) .* (sv - ss_amt));
}
