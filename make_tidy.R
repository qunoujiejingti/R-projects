library(tidyverse)
library(tidymodels)

source("functions.R")

# The function that will be contained in the parsnip package
logistic_lasso <- function(mode = "classification", penalty){
  args <- list(penalty = rlang::enquo(penalty))
  new_model_spec("logistic_lasso",
                 args = args,
                 mode = mode,
                 eng_args = NULL,
                 method = NULL,
                 engine = NULL)
}

# Registering the model
set_new_model("logistic_lasso")
set_model_mode(model = "logistic_lasso", mode = "classification")
set_model_engine("logistic_lasso",
                 mode = "classification",
                 eng = "fit_logistic_lasso")
set_dependency("logistic_lasso", eng = "fit_logistic_lasso", pkg = "base")

# Tell parsnip what the parameters are in the model
set_model_arg(
  model = "logistic_lasso",
  eng = "fit_logistic_lasso",
  parsnip = "penalty",
  original = "lambda",
  func = list(pkg = "dials", fun = "penalty"),
  has_submodel = FALSE
)

# Tell parsnip how to go from a formula to the x matrix
set_encoding(
  model = "logistic_lasso",
  eng = "fit_logistic_lasso",
  mode = "classification",
  options = list(
    predictor_indicators = "traditional",
    compute_intercept = TRUE,
    remove_intercept = FALSE,
    allow_sparse_x = FALSE
    )
)

# Show model info
show_model_info("logistic_lasso")

# Now we tell parsnip how to fit the model and do prediction
set_fit(
  model = "logistic_lasso",
  eng = "fit_logistic_lasso",
  mode = "classification",
  value = list(
    interface = "matrix",
    protect = c("x", "y"),
    func = c(fun = "fit_logistic_lasso"),
    defaults = list()
  )
)

set_pred(
  model = "logistic_lasso",
  eng = "fit_logistic_lasso",
  mode = "classification",
  type = "class",
  value = list(
    pre = NULL,
    post = NULL,
    func = c(fun = "predict_logistic_lasso"),
    args = list(
      object = expr(object$fit),
      new_x = expr(as.matrix(new_data[, names(object$fit$beta)]))
    )
  )
)

predict_logistic_lasso_prob <- function(object, new_x){
  new_x <- cbind(1, new_x)
  beta <- c(object$intercept, object$beta)
  logit_p = (new_x %*% beta) %>% as.numeric
  return( 1/(1 + exp(-logit_p) ))
}

set_pred(
  model = "logistic_lasso",
  eng = "fit_logistic_lasso", 
  mode = "classification",
  type = "prob",
  value = list(
    pre = NULL,
    post = NULL,
    func = c(fun = "predict_logistic_lasso_prob"),
    args = list(
      fit = expr(object$fit),
      new_x = expr(as.matrix(new_data[, names(object$fit$beta)]))
    )
  )
)

# Prepare for finalized workflow after tuning the model
update.logistic_lasso <- function(object, penalty = NULL){
  if(!is.null(penalty)){
    object$args <- list(penalty = enquo(penalty))
  }
  new_model_spec("logistic_lasso", args = object$args, eng_args = NULL,
                 mode = "classification", method = NULL, engine = object$engine)
}