fit_logistic_lasso <- function(x, y, lambda, beta0 = NULL, eps = 0.0001, iter_max = 100){
  ### Description: This function is an algorithm realization of logistic regression Lasso
  ### with respect to coordinate descent method in updating the coefficients beta.
  ###
  ### Input:
  ### -x: matrix of predictors (not including the intercept)
  ### -y: vector of data
  ### -lambda: penalty (tuning parameter)
  ### -beta0: initial guess
  ### -eps: parameter for stopping criterion
  ### -iter_max: maximum number of iterations
  ###
  ### Output:
  ### -ret: A list containing the members intercept, beta and lambda
  ###
  ### Example:
  ### x,y <- simulated data
  ### ret <- fit_logistic_lasso(x=x, y=y, lambda = 0.3)
  
  x <- cbind(1, x) # we add the intercept and update it with coordinate descent
  colnames(x)[1] = '(intercept)'
  p <- dim(x)[2]
  n <- dim(x)[1]
  
  # Initilialize the parameter vector beta
  if(is.null(beta0)){
    beta0 <- rep(0,p)
  }
  
  # Transform y into numerical vector
  fct_levels <- levels(y)
  y <- as.numeric(y) - 1
  
  # Compute the logistic function
  beta <- beta0
  x_beta <- as.numeric(x %*% beta)
  px <- 1/(1 + exp(-x_beta))
  
  for (iter in 1:iter_max) {
    # Update the weights and working response respectively
    w <- px * (1 - px)
    z <- x_beta + (y - px)/w
    
    # Update the jth beta entry by coordinate descent
    for (j in 1:p){
      # Compute the jth partial residual
      rj <- z - x[,-j] %*% beta[-j]
      
      # Compute the soft-threshold
      sign_of_number <- sign(sum(w*rj*x[,j]))
      abs_of_number_minus_lambda <- abs(sum(w*rj*x[,j])) - lambda
      
      # Update
      beta[j] <-  sign_of_number * (max(abs_of_number_minus_lambda, 0)) / sum(w*x[,j]^2)
    }
    
    # Update the value of x times beta and the value of logistic function
    names(beta) <- colnames(x)
    x_beta <- as.numeric(x %*% beta)
    px <- 1/(1 + exp(-x_beta))
    
    # Check if converge
    if(max(abs(beta - beta0))/max(abs(beta)) < eps){
      # Converged, return what asked
      # intercept <- mean(y) - sum(beta * x_bar)
      return(list(intercept = beta[1], beta = beta[-1], lambda = lambda, converged = TRUE,
                  fct_levels = fct_levels))
    }
    beta0 <- beta
  }
  warning(paste("Method did not converge in", iter_max, "iterations", sep=" "))
  # intercept <- mean(y) - sum(beta * x_bar)
  return(list(intercept = beta[1], beta = beta[-1], lambda = lambda, converged = FALSE,
              fct_levels = fct_levels))
}

predict_logistic_lasso <- function(object, new_x){
  ### Description: This function does prediction job of logistic regression, it takes the output
  ### from the fit_logistic_lasso and some test data (maybe more than one point) and outputs the
  ### prediction result.
  ###
  ### Input:
  ### -object: Output from fit_logistic_lasso
  ### -new_x: Data to predict at (may be more than one point)
  ###
  ### Output: 
  ### -ret: The prediction results in factor form
  ### Example:
  ### logistic_lasso_result <- fit_logistic_lasso(x=x,y=y,lambda=0.3)
  ### predict(logistic_lasso, new_data = test) %>% bind_cols(test %>% select(y)) %>%
  ### conf_mat(truth = y, estimate = .pred_class) (This will produce the prediction results vs
  ### true results in a matrix form)
  
  new_x <- cbind(1, new_x)
  beta  <- c(object$intercept, object$beta)
  numeric_pred <- ((new_x %*% beta) >= 0) %>% as.numeric
  return( object$fct_levels[numeric_pred + 1] %>% factor )
}