// Rung 0: simple statistics. Chance of "later" from reward size and delay. No psychology.
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
transformed data {
  vector[N] x_amt = log(amt / ss_amt);     // how much bigger the later reward is (log ratio)
  vector[N] x_del = log(delay);            // log delay
}
parameters {
  vector[3] mu;                    // group averages
  vector<lower=0>[3] sigma_p;      // spread between people
  vector<lower=0>[3] sigma_s;      // spread between a person's two sessions
  matrix[P, 3] theta_p;            // each person's average traits
  matrix[U, 3] z_s;                // session offsets (standardised)
}
transformed parameters {
  matrix[U, 3] theta = theta_p[person, ] + diag_post_multiply(z_s, sigma_s);  // intercept, amount slope, delay slope
}
model {
  mu[1] ~ normal(0, 3); mu[2] ~ normal(0, 3); mu[3] ~ normal(0, 3);
  sigma_p ~ normal(0, 1);
  sigma_s ~ normal(0, 0.5);
  for (k in 1:3) theta_p[, k] ~ normal(mu[k], sigma_p[k]);
  to_vector(z_s) ~ std_normal();
  choice ~ bernoulli_logit(theta[unit, 1] + theta[unit, 2] .* x_amt + theta[unit, 3] .* x_del);
}
