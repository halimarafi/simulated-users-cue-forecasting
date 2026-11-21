// Forecast C, rung 2: how much does the cue shift patience, log(k), and caution, log(boundary), on average?
// All other traits stay at each person's basic-trial values.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  vector<lower=0>[N] rt;
  vector<lower=0>[N] amt;
  vector<lower=0>[N] delay;
  real<lower=0> ss_amt;
  matrix[N, 6] th;                 // fixed traits per trial (same order as rung 2)
  vector<lower=0>[N] tau;          // fixed non-decision time per trial (seconds)
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
  vector[N] bias = inv_logit(th[, 4]);
  vector[N] vmax = exp(th[, 5]);
  vector[N] vcoef = exp(th[, 6]);
}
parameters {
  // Limits far outside any realistic cue effect (earlier studies: about -0.2 on log k). They stop the sampler's
  // early exploratory steps from reaching values where the boundary overflows to infinity.
  real<lower=-2, upper=2> d_logk;
  real<lower=-1, upper=1> d_log_alpha;
}
model {
  d_logk ~ normal(0, 1);
  d_log_alpha ~ normal(0, 0.5);
  vector[N] alpha = exp(th[, 2] + d_log_alpha);
  vector[N] sv = amt ./ (1 + exp(th[, 1] + d_logk) .* delay);
  vector[N] v = vmax .* (2 * inv_logit(vcoef .* (sv - ss_amt)) - 1);
  rt[i1] ~ wiener(alpha[i1], tau[i1], bias[i1], v[i1]);
  rt[i0] ~ wiener(alpha[i0], tau[i0], 1 - bias[i0], -v[i0]);
}
