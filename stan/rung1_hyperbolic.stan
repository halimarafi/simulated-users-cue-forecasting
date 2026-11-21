// Rung 1: patience model. Hyperbolic discounting with a softmax choice rule.
data {
  int<lower=1> N;                          // trials
  int<lower=1> P;                          // people
  int<lower=1> U;                          // person-sessions ("units")
  array[U] int<lower=1, upper=P> person;   // which person each unit belongs to
  array[N] int<lower=1, upper=U> unit;     // unit index per trial
  array[N] int<lower=0, upper=1> choice;   // 1 = later, 0 = now
  vector<lower=0>[N] amt;                  // later amount
  vector<lower=0>[N] delay;                // delay in days
  real<lower=0> ss_amt;                    // immediate amount (20)
}
parameters {
  vector[2] mu;                    // group averages
  vector<lower=0>[2] sigma_p;      // spread between people
  vector<lower=0>[2] sigma_s;      // spread between a person's two sessions
  matrix[P, 2] theta_p;            // each person's average traits
  matrix[U, 2] z_s;                // session offsets (standardised)
}
transformed parameters {
  matrix[U, 2] theta = theta_p[person, ] + diag_post_multiply(z_s, sigma_s);
  vector[U] logk = theta[, 1];          // patience (log discount rate; lower = more patient)
  vector[U] beta = exp(theta[, 2]);     // choice consistency
}
model {
  mu[1] ~ normal(-4, 2);
  mu[2] ~ normal(-1, 1);
  sigma_p ~ normal(0, 1);
  sigma_s ~ normal(0, 0.5);
  for (k in 1:2) theta_p[, k] ~ normal(mu[k], sigma_p[k]);
  to_vector(z_s) ~ std_normal();
  vector[N] sv = amt ./ (1 + exp(logk[unit]) .* delay);
  choice ~ bernoulli_logit(beta[unit] .* (sv - ss_amt));
}
