data {
  int<lower=0> tot_T;
  int<lower=0> T;
  int<lower=0> p;
  matrix[T,p] x;
  array[T] int<lower=0, upper=1> y;
  vector[p+1] prechange_mean;
  matrix[p+1,p+1] prechange_cov;
  vector[p+1] shift_mean;
  matrix[p+1,p+1] shift_cov;
  vector[T] prior_change_probs;
}

parameters {
  vector[p+1] theta0;
  vector[p+1] theta1;
}

transformed parameters {
    vector[T] lp;
    {
      vector[T+1] lp_pre;
      vector[T+1] lp_post;
      lp_pre[1] = 0;
      lp_post[1] = 0;
      // Use cumulative Sums??
      for (t in 1:T) {
        lp_pre[t + 1] = lp_pre[t] + bernoulli_logit_lpmf(y[t] | dot_product(x[t], theta0[1:p]) + theta0[p+1]);
        lp_post[t + 1] = lp_post[t] + bernoulli_logit_lpmf(y[t] | dot_product(x[t], theta1[1:p]) + theta1[p+1]);
      }
      lp[1:T-1] = log(prior_change_probs[1:T-1])
          + rep_vector(lp_post[T + 1], T-1)
          + lp_pre[2:T] - lp_post[2:T];
      lp[T] = log(prior_change_probs[T]) + lp_pre[T+1];
    }
}

model {
  theta0 ~ multi_normal(prechange_mean, prechange_cov);
  theta1 ~ multi_normal(theta0 + shift_mean, shift_cov);

  target += log_sum_exp(lp);
}
