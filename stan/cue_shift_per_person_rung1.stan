// Notebook 06: how much did the cue shift each person's patience, log(k), on each day?
// Each person-day's basic-trial traits are fixed (notebook 02); one cue shift per person-day is estimated.
// The shifts are tied together by a group average and spread, which keeps noisy person-days from extreme values.
data {
  int<lower=1> N;                          // cued trials
  int<lower=1> U;                          // person-days
  array[N] int<lower=1, upper=U> unit;
  array[N] int<lower=0, upper=1> choice;
  vector<lower=0>[N] amt;
  vector<lower=0>[N] delay;
  real<lower=0> ss_amt;
  vector[U] logk;                          // fixed basic-trial patience per person-day
  vector[U] log_beta;                      // fixed basic-trial choice consistency per person-day
}
parameters {
  real<lower=-2, upper=2> mu;              // group average shift
  real<lower=0, upper=2> sigma;            // spread of shifts between person-days
  vector[U] z;
}
transformed parameters {
  vector[U] shift = mu + sigma * z;        // each person-day's shift in log(k); negative = more patient with the cue
}
model {
  mu ~ normal(0, 1);
  sigma ~ normal(0, 0.5);
  z ~ std_normal();
  vector[N] sv = amt ./ (1 + exp(logk[unit] + shift[unit]) .* delay);
  choice ~ bernoulli_logit(exp(log_beta[unit]) .* (sv - ss_amt));
}
