rm(list = ls()); gc(reset = TRUE)
script_file_path <- rstudioapi::getSourceEditorContext()$path
setwd(script_file_path)
your_python_path <- "/Users/someone/miniconda3/envs/seek/bin/python3.8"
solve.sta <- function(P, rational=FALSE){
  n <- nrow(P)
  y <- c(rep(0,n), 1)
  X <- rbind(t(P) - diag(n), 1)
  p <- solve(crossprod(X), crossprod(X, y)) |> c()
  if(rational){
    MASS::fractions(p)
  } else {
    p
  }
}

P <- matrix(1 / 7, nrow = 8, ncol = 8)
diag(P) <- 0.0
# solve.sta(P)

library(markovchain)
mcB <- new("markovchain", transitionMatrix = P)

beta_mix_func <- function(x) {
  y <- (7 / 8) * 7^(-x)
  return(y)
}
find_argmin <- function(beta_mix_func, K_max, threshold) {
  find <- FALSE
  for(k in 1:K_max) {
    if (beta_mix_func(k) < k * threshold) {
      find <- TRUE
      break
    }
  }
  if (find) {
    return(k)
  } else {
    return(NA)
  }
}

select_K <- function(beta_mix, NT, delta=0.05) {
  write.table(beta_mix, "est_beta.txt", col.names = FALSE, row.names = FALSE)
  call_template <- "%s %s/K_consistency.py --NT=%i --delta=%s"
  system(sprintf(call_template, your_python_path, dirname(script_file_path), NT, as.character(delta)))
  return(as.vector(read.table("select_K.txt"))[[1]])
}

library(dplyr)
library(progress)
sample_list <- c(2^9, 2^12, 2^15, 2^18, 2^21, 2^24)
delta_list <- c(0.01, 0.05, 0.1)
n_burnin <- 5000
nrep <- 20
K_gap_list <- list()
for (delta in delta_list) {
  K_hat <- matrix(nrow = length(sample_list), ncol = nrep, data = 0)
  pb <- progress_bar$new(total = nrep)
  K_star_list <- numeric(length(sample_list))
  for (i in 1:length(sample_list)) {
    K_star_list[i] <- find_argmin(beta_mix_func, K_max = 15, 
                                  threshold = delta / sample_list[i])
  }
  for (r in 1:nrep) {
    # generate the sequence
    set.seed(r)
    max_NT <- max(sample_list)
    outs <- markovchainSequence(n = max_NT + n_burnin, 
                                markovchain = mcB)[-(1:n_burnin)]
    for (i in 1:length(sample_list)) {
      NT <- sample_list[i]
      outs_NT <- outs[(max_NT - NT + 1):max_NT]
      
      # marginal probability
      m_distr <- prop.table(table(outs_NT))
      prod_m_distr <- expand.grid("x1" = names(m_distr), "x2" = names(m_distr))
      prod_m_distr[["emp_prop"]] <- apply(expand.grid(m_distr, m_distr), 1, prod)
      
      max_K <- 15
      K_range <- 1:max_K
      beta_mix <- numeric(max_K)
      for (K in K_range) {
        # joint probability
        j_distr <- data.frame(cbind(outs_NT, lag(outs_NT, n = K)))[-(1:K), ]
        colnames(j_distr) <- c("x1", "x2")
        j_distr[["x1"]] <- factor(j_distr[["x1"]])
        j_distr[["x2"]] <- factor(j_distr[["x2"]])
        j_distr <- data_frame(j_distr) %>% 
          count(x1, x2, .drop = FALSE) %>% 
          arrange(x2, x1)
        colnames(j_distr)[3] <- "emp_prop"
        j_distr[["emp_prop"]] <- j_distr[["emp_prop"]] / sum(j_distr[["emp_prop"]])
        j_distr <- as.data.frame(j_distr)
        # beta mixing coefficients
        beta_mix[K] <- 0.5 * sum(abs(j_distr[["emp_prop"]] - prod_m_distr[["emp_prop"]]))
      }
      K_hat[i, r] <- select_K(beta_mix, NT, delta)
    }
    i <- i + 1
    pb$tick()
  }
  print(K_hat)
  K_gap <- K_hat - K_star_list 
  K_gap_list[[as.character(delta)]] <- K_gap
}

# save(K_gap_list, file = "K_est.rda")
load("K_est.rda")
K_gap_list <- lapply(K_gap_list, function(x) {
  x / K_star_list
})
K_hat_0.01 <- K_gap_list[[1]]  # save the results when delta = 0.01
K_hat_0.05 <- K_gap_list[[2]]  # save the results when delta = 0.05
K_hat_0.1 <- K_gap_list[[3]]  # save the results when delta = 0.01


library(ggplot2)
library(reshape2)
rownames(K_hat_0.01) <- rownames(K_hat_0.05) <- rownames(K_hat_0.1) <- sample_list
pdat1 <- melt(t(K_hat_0.05))
pdat1[["delta"]] <- 0.05
pdat2 <- melt(t(K_hat_0.01))
pdat2[["delta"]] <- 0.01
pdat3 <- melt(t(K_hat_0.1))
pdat3[["delta"]] <- 0.1
pdat <- rbind.data.frame(pdat1, pdat2, pdat3)
# pdat <- subset(pdat, value >= 0)
pdat[["bias"]] <- pdat[["value"]]
pdat[["mse"]] <- (pdat[["value"]])^2
pdat <- data_frame(pdat) %>% group_by(delta, Var2) %>% 
  summarise(Bias = mean(bias), MSE = mean(mse))
pdat <- melt(pdat, id.vars = c("delta", "Var2"))
pdat[["delta"]] <- factor(pdat[["delta"]])
p <- ggplot(pdat, aes(x = log2(Var2), y = value, 
                      color = delta, shape = delta, linetype = delta)) + 
  facet_wrap(variable ~ ., scales = "free") + 
  geom_point(size = 3.0) + 
  geom_line(size = 1.2) + 
  scale_x_continuous("log(NT)") + 
  # scale_x_continuous("NT", trans = "log2", 
  #                    labels = scales::label_log(base = 2)) + 
  scale_color_discrete("Delta") + 
  scale_shape_discrete("Delta") + 
  scale_linetype_discrete("Delta") + 
  scale_y_continuous("", trans = "pseudo_log", n.breaks = 5) + 
  theme_bw() + 
  theme(legend.position = c(0.9, 0.8), 
        legend.background = element_rect(colour = "black"))
p
ggsave(p, filename = "binary_independent.pdf", width = 8, height = 3.8)