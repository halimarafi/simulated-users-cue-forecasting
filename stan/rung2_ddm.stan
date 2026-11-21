// Rung 2: patience + decision time. Temporal-discounting drift-diffusion model.
// Upper boundary = "later", lower boundary = "now". Drift depends on the value
// difference through an S-shaped function (Peters & D'Esposito 2020; Brands et al. 2024).
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
  vector<lower=0>[N] rt;                   // response time (s)
  vector<lower=0>[U] min_rt;               // fastest kept response per unit
}
transformed data {
  int N1 = sum(choice);
  int N0 = N - N1;
  array[N1] int i1;
  array[N0] int i0;
  {
    int a = 0; int b = 0;
    for (n in 1:N) {
      if (choice[n] == 1) { a += 1; i1[a] = n; } else { b += 1; i0[b] = n; }
    }
  }
}
parameters {
  vector[6] mu;                    // group averages
  vector<lower=0>[6] sigma_p;      // spread between people
  vector<lower=0>[6] sigma_s;      // spread between a person's two sessions
  matrix[P, 6] z_p;                // person offsets (standardised)
  matrix[U, 6] z_s;                // session offsets (standardised)
}
transformed parameters {
  // Written as "group average + spread x standardised offset" at both levels. This is the
  // same model, but the sampler moves through it far more easily (no "funnel").
  matrix[P, 6] theta_p = rep_matrix(mu', P) + diag_post_multiply(z_p, sigma_p);   // each person's average traits
  matrix[U, 6] theta = theta_p[person, ] + diag_post_multiply(z_s, sigma_s);
  vector[U] logk  = theta[, 1];                        // patience (log discount rate)
  vector[U] alpha = exp(theta[, 2]);                   // caution (boundary separation)
  // Non-decision time can reach at most 95% of the person's fastest response. At 100% the
  // fastest trial becomes impossible and the sampler gets stuck at that edge.
  vector[U] tau   = 0.95 * min_rt .* inv_logit(theta[, 3]);   // non-decision time
  vector[U] bias  = inv_logit(theta[, 4]);             // starting point (0.5 = no lean)
  vector[U] vmax  = exp(theta[, 5]);                   // maximum drift
  vector[U] vcoef = exp(theta[, 6]);                   // how sharply drift reacts to value
}
model {
  mu[1] ~ normal(-4, 2);
  mu[2] ~ normal(0.7, 0.5);
  mu[3] ~ normal(0, 1);
  mu[4] ~ normal(0, 0.5);
  mu[5] ~ normal(0.7, 0.5);
  mu[6] ~ normal(-2.3, 1);
  sigma_p ~ normal(0, 1);
  sigma_s ~ normal(0, 0.5);
  to_vector(z_p) ~ std_normal();
  to_vector(z_s) ~ std_normal();
  vector[N] sv = amt ./ (1 + exp(logk[unit]) .* delay);
  vector[N] v  = vmax[unit] .* (2 * inv_logit(vcoef[unit] .* (sv - ss_amt)) - 1);
  rt[i1] ~ wiener(alpha[unit[i1]], tau[unit[i1]], bias[unit[i1]], v[i1]);
  rt[i0] ~ wiener(alpha[unit[i0]], tau[unit[i0]], 1 - bias[unit[i0]], -v[i0]);
}
