data {
  int<lower=0> tot_T;
  int<lower=0> T;
  int<lower=0> p;
  int<lower=0> delta_p;
  matrix[T,p] x;
  matrix[T,delta_p] delta_x;
  array[T] int<lower=0, upper=1> y;
  vector[p+1] prechange_mean;
  matrix[p+1,p+1] prechange_cov;
  matrix[delta_p+1,delta_p+1] shift_cov;
  vector[T] prior_change_probs;
}

parameters {
  vector[p+1] theta0;
  vector[delta_p+1] theta1;
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
        real orig_prob = 1/(1 + exp(-(dot_product(x[t], theta0[1:p]) + theta0[p+1])));
        real change_risk = dot_product(delta_x[t], theta1[1:delta_p]) + theta1[delta_p+1];
        real new_prob = fmax(0.00001, fmin(1 - 0.00001, orig_prob + change_risk));
        real new_logit = log(new_prob/(1 - new_prob));
        lp_post[t + 1] = lp_post[t] + bernoulli_logit_lpmf(y[t] | new_logit);
      }
      lp[1:T-1] = log(prior_change_probs[1:T-1])
          + rep_vector(lp_post[T + 1], T-1)
          + lp_pre[2:T] - lp_post[2:T];
      lp[T] = log(prior_change_probs[T]) + lp_pre[T+1];
    }
}

model {
  theta0 ~ multi_normal(prechange_mean, prechange_cov);
  theta1 ~ multi_normal(rep_vector(0,delta_p+1), shift_cov);

  target += log_sum_exp(lp);
}
