# Data/Scripts/vb_models_baseline.R
# Baselines for NC State Volleyball — CV-only (repeated 5-fold, 10 repeats)
# Outputs: CSV + PNGs in Data/Processed
#
# Models:
#   - Logistic Regression (glm)
#   - kNN (k = 3, kknn)
#   - Decision Tree (rpart; depth=4, min_n=5)
#   - Random Forest (ranger; trees=2000, mtry=4, min_n=5, num.threads=1)
#
# NOTE: Holdout evaluation is disabled for course submission.
#       After the season, you can re-enable it (see stub near bottom).

# ---------- packages ----------
required <- c("tidyverse","tidymodels","ranger","vip","pROC","kknn")
to_install <- required[!required %in% rownames(installed.packages())]
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org", dependencies = TRUE)
}

library(tidyverse)
library(tidymodels)
library(ranger)
library(vip)
library(pROC)

# ---------- reproducibility ----------
RNGkind("L'Ecuyer-CMRG")
set.seed(563)
if (requireNamespace("foreach", quietly = TRUE)) foreach::registerDoSEQ()
options(mc.cores = 1)

# ---------- paths ----------
args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- NULL
for (a in args) if (startsWith(a, file_arg)) script_path <- substring(a, nchar(file_arg) + 1)
SCRIPT_DIR <- if (is.null(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
PROCESSED  <- file.path(SCRIPT_DIR, "..", "Processed")

train_path <- file.path(PROCESSED, "ncsu_volleyball_train_through_2025-10-11.csv")
# Test/holdout path (optional). If this file exists the script will perform a holdout
# evaluation: fit final models on all TRAIN and evaluate on the TEST rows with known outcomes.
test_path  <- file.path(PROCESSED, "ncsu_volleyball_holdout_after_2025-10-11.csv")

dir.create(PROCESSED, showWarnings = FALSE, recursive = TRUE)

# ---------- outputs ----------
out_metrics_csv    <- file.path(PROCESSED, "model_metrics.csv")
out_rf_vi_csv      <- file.path(PROCESSED, "rf_variable_importance.csv")
out_preds_csv      <- file.path(PROCESSED, "predictions_summary.csv")
out_auc_bar_png    <- file.path(PROCESSED, "model_auc_bar.png")
out_roc_png        <- file.path(PROCESSED, "model_roc_curves.png")
out_rf_vi_png      <- file.path(PROCESSED, "rf_variable_importance.png")
out_rf_vi_full_csv <- file.path(PROCESSED, "rf_variable_importance_full.csv")
out_rf_vi_full_png <- file.path(PROCESSED, "rf_variable_importance_full.png")
out_cm_png         <- file.path(PROCESSED, "confusion_matrix_heatmap.png") # (only used if you re-enable holdout)

# ---------- load data ----------
train <- read.csv(train_path)

# ---------- features & cleaning ----------
feature_cols <- c(
  "sets","kills","errors","total_attacks","hitting_pct","assists","aces",
  "serve_errors","digs","receive_attempts","receive_errors",
  "block_solos","block_assists","block_errors","points","ball_handling_errors"
)

numfix <- function(df) {
  df %>%
    mutate(across(all_of(feature_cols), function(x) suppressWarnings(as.numeric(x)))) %>%
    mutate(across(all_of(feature_cols), ~ replace(., is.na(.), 0)))
}

train <- numfix(train)
train$win <- factor(train$win, levels = c(0,1))

# ---------- recipe ----------
rec <- recipe(win ~ ., data = train %>% select(all_of(feature_cols), win)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# ---------- models (fixed, no tuning) ----------
# labels kept stable for CSV/plots/report
log_label  <- "Logistic Regression (engine=glm)"
knn_label  <- "kNN (k=3)"
tree_label <- "Decision Tree (depth=4,min_n=5)"
rf_label   <- "Random Forest (trees=2000,mtry=4,min_n=5)"

mod_log  <- logistic_reg(mode = "classification") %>% set_engine("glm")
mod_knn  <- nearest_neighbor(mode = "classification", neighbors = 3) %>% set_engine("kknn")
mod_tree <- decision_tree(mode = "classification", tree_depth = 4, min_n = 5) %>% set_engine("rpart")
mod_rf   <- rand_forest(mode = "classification", trees = 2000, mtry = 4, min_n = 5) %>%
            set_engine("ranger", importance = "impurity", probability = TRUE, num.threads = 1)

wf_log  <- workflow() %>% add_model(mod_log)  %>% add_recipe(rec)
wf_knn  <- workflow() %>% add_model(mod_knn)  %>% add_recipe(rec)
wf_tree <- workflow() %>% add_model(mod_tree) %>% add_recipe(rec)
wf_rf   <- workflow() %>% add_model(mod_rf)   %>% add_recipe(rec)

# ---------- metrics & plotting helpers ----------
metric_set_use <- metric_set(accuracy, kap, roc_auc)

save_auc_bar <- function(metrics_tbl, mode_label) {
  p <- metrics_tbl %>%
    arrange(roc_auc) %>%
    mutate(model = factor(model, levels = model)) %>%
    ggplot(aes(x = model, y = roc_auc)) +
    geom_col(width = 0.6) +
    coord_flip() +
    labs(title = paste0("Model AUC (", mode_label, ")"),
         x = NULL, y = "ROC AUC") +
    theme_minimal(base_size = 12)
  ggsave(out_auc_bar_png, p, width = 6, height = 4, dpi = 150)
}

save_roc_curves_from_cv <- function(preds_log_cv, preds_rf_cv, title_label = "ROC Curves (Repeated 5-fold CV)") {
  if (is.null(preds_log_cv) || is.null(preds_rf_cv)) return(invisible(NULL))
  if (length(unique(preds_log_cv$win)) < 2 && length(unique(preds_rf_cv$win)) < 2) return(invisible(NULL))

  roc_log <- tryCatch(roc_curve(preds_log_cv, truth = win, .pred_1) %>% mutate(model = "Logistic"), error = function(e) NULL)
  roc_rf  <- tryCatch(roc_curve(preds_rf_cv,  truth = win, .pred_1) %>% mutate(model = "Random Forest"), error = function(e) NULL)
  rc_all  <- bind_rows(roc_log, roc_rf)
  if (nrow(rc_all) == 0) return(invisible(NULL))

  p <- rc_all %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
    geom_path(size = 1) +
    geom_abline(linetype = 2) +
    coord_equal() +
    labs(title = title_label, x = "False Positive Rate", y = "True Positive Rate") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  ggsave(out_roc_png, p, width = 6, height = 5, dpi = 150)
}

# =========================================================
# CV-ONLY (Repeated 5-fold; 10 repeats)  — used for the project
# =========================================================
message("Mode: REPEATED 5-FOLD CV (10 repeats) on TRAIN.")

# Save a snapshot of TRAIN for reproducibility (optional but handy)
snapshot_path <- file.path(PROCESSED, paste0("train_snapshot_", format(Sys.time(), "%Y%m%dT%H%M%S"), ".csv"))
write_csv(train, snapshot_path)

set.seed(563)
folds <- vfold_cv(train, v = 5, repeats = 10, strata = win)
res_control <- control_resamples(save_pred = TRUE)

set.seed(563); res_log  <- fit_resamples(wf_log,  resamples = folds, metrics = metric_set_use, control = res_control)
set.seed(563); res_knn  <- fit_resamples(wf_knn,  resamples = folds, metrics = metric_set_use, control = res_control)
set.seed(563); res_tree <- fit_resamples(wf_tree, resamples = folds, metrics = metric_set_use, control = res_control)
set.seed(563); res_rf   <- fit_resamples(wf_rf,   resamples = folds, metrics = metric_set_use, control = res_control)

summarize_metrics <- function(rs, name) {
  collect_metrics(rs) %>%
    select(.metric, mean) %>%
    pivot_wider(names_from = .metric, values_from = mean) %>%
    mutate(model = name)
}

results <- bind_rows(
  summarize_metrics(res_log,  log_label),
  summarize_metrics(res_knn,  knn_label),
  summarize_metrics(res_tree, tree_label),
  summarize_metrics(res_rf,   rf_label)
) %>%
  mutate(
    accuracy = coalesce(accuracy, NA_real_),
    kap      = coalesce(kap, NA_real_),
    roc_auc  = coalesce(roc_auc, NA_real_)
  ) %>%
  select(model, accuracy, kap, roc_auc) %>%
  mutate(mode = "CV") %>%
  relocate(model, mode) %>%
  # enforce your preferred order
  mutate(model = factor(model, levels = c(log_label, knn_label, tree_label, rf_label))) %>%
  arrange(model) %>%
  mutate(across(c(accuracy, kap, roc_auc), ~ round(., 3)))

write_csv(results, out_metrics_csv)
save_auc_bar(results, "Repeated 5-fold CV (10 repeats)")

# ROC curves from CV predictions (Logistic vs Random Forest)
safe_collect_preds <- function(rs) {
  tryCatch(collect_predictions(rs) %>% select(id, .row, win, .pred_1 = .pred_1), error = function(e) NULL)
}
preds_log_cv <- safe_collect_preds(res_log)
preds_rf_cv  <- safe_collect_preds(res_rf)
save_roc_curves_from_cv(preds_log_cv, preds_rf_cv, "ROC Curves (Repeated 5-fold CV)")

# RF variable importance: fit RF on ALL train, save CSV + PNG
fit_rf_full <- fit(wf_rf, data = train)
rf_fit <- extract_fit_parsnip(fit_rf_full)$fit
vi_tbl <- enframe(rf_fit$variable.importance, name = "feature", value = "importance") %>%
  arrange(desc(importance))
write_csv(vi_tbl, out_rf_vi_csv)
ggsave(out_rf_vi_png, vip(rf_fit, num_features = min(12, length(feature_cols))), width = 7, height = 5, dpi = 150)

# CV predictions summary (mean prob by repeat & truth) — matches your table
rf_cv_preds <- tryCatch(collect_predictions(res_rf), error = function(e) NULL)
if (!is.null(rf_cv_preds)) {
  preds_summary <- rf_cv_preds %>%
    transmute(id, truth = win, prob_win = .pred_1) %>%
    group_by(id, truth) %>%
    summarise(mean_prob = mean(prob_win, na.rm = TRUE), .groups = "drop")
  write_csv(preds_summary, out_preds_csv)
}

# =========================================================
# HOLDOUT EVALUATION (disabled for the course submission)
# =========================================================
if (file.exists(test_path)) {
  message("Test file found — performing HOLDOUT evaluation on TEST set.")
  test <- read.csv(test_path)
  test <- numfix(test)
  test_eval <- test %>% dplyr::filter(!is.na(win))

  if (nrow(test_eval) > 0) {
    test_eval$win <- factor(test_eval$win, levels = c(0,1))

    # Fit final models on the full TRAIN dataset
    fit_log_full  <- fit(wf_log,  data = train)
    fit_knn_full  <- fit(wf_knn,  data = train)
    fit_tree_full <- fit(wf_tree, data = train)
    fit_rf_full2  <- fit(wf_rf,   data = train)

    # Helper to produce predictions (class + prob) and compute metrics
    make_preds <- function(fit, data) {
      prob <- predict(fit, data, type = "prob")
      cl   <- predict(fit, data)
      bind_cols(data %>% select(win), cl, prob)
    }

    preds_log_hold <- make_preds(fit_log_full, test_eval)
    preds_knn_hold <- make_preds(fit_knn_full, test_eval)
    preds_tree_hold <- make_preds(fit_tree_full, test_eval)
    preds_rf_hold  <- make_preds(fit_rf_full2, test_eval)

    calc_hold_metrics <- function(preds, name) {
      acc <- tryCatch(accuracy(preds, truth = win, estimate = .pred_class) %>% pull(.estimate), error = function(e) NA_real_)
      kapv <- tryCatch(kap(preds, truth = win, estimate = .pred_class) %>% pull(.estimate), error = function(e) NA_real_)
      roc  <- tryCatch(roc_auc(preds, truth = win, .pred_1) %>% pull(.estimate), error = function(e) NA_real_)
      tibble(model = name, accuracy = acc, kap = kapv, roc_auc = roc)
    }

    holdout_results <- bind_rows(
      calc_hold_metrics(preds_log_hold,  log_label),
      calc_hold_metrics(preds_knn_hold,  knn_label),
      calc_hold_metrics(preds_tree_hold, tree_label),
      calc_hold_metrics(preds_rf_hold,   rf_label)
    ) %>%
      mutate(mode = "HOLDOUT") %>%
      relocate(model, mode) %>%
      mutate(across(c(accuracy, kap, roc_auc), ~ round(as.numeric(.), 3)))

    # Write combined metrics (CV + HOLDOUT)
    combined_metrics <- bind_rows(results, holdout_results)
    write_csv(combined_metrics, out_metrics_csv)

    # Write per-match predictions summary for HOLDOUT (mean prob per match not needed — use per-row)
    preds_all_hold <- bind_rows(
      preds_log_hold  %>% mutate(model = log_label),
      preds_knn_hold  %>% mutate(model = knn_label),
      preds_tree_hold %>% mutate(model = tree_label),
      preds_rf_hold   %>% mutate(model = rf_label)
    ) %>%
      mutate(row_id = row_number()) %>%
      select(row_id, model, truth = win, .pred_class, .pred_1)
    write_csv(preds_all_hold, out_preds_csv)

    # Confusion matrix + ROC for Random Forest (example)
    tryCatch({
      cm_rf <- conf_mat(preds_rf_hold, truth = win, estimate = .pred_class)
      png(out_cm_png, width = 800, height = 600)
      autoplot(cm_rf) + ggtitle("Confusion Matrix (Random Forest on Holdout)")
      dev.off()
    }, error = function(e) message("Could not create confusion matrix plot: ", e$message))

    tryCatch({
      save_roc_curves_from_cv(preds_log_hold, preds_rf_hold, title_label = "ROC Curves (Holdout)")
    }, error = function(e) message("Could not save holdout ROC curves: ", e$message))
  } else {
    message("Test file present but no rows with known outcomes (no holdout evaluation performed).")
  }
} else {
  message("No test file found at: ", test_path, " — skipping holdout evaluation.")
}

# ---------- console summary ----------
if (file.exists(out_metrics_csv)) {
  cat("\nSaved:\n", out_metrics_csv, "\n", out_rf_vi_csv, "\n",
      out_preds_csv, "\n", out_auc_bar_png, "\n", out_roc_png, "\n",
      out_rf_vi_png, "\n", sep = "")
}

# Fit overall best model (Random Forest) on entire dataset
if (file.exists(test_path)) {
  full_data <- bind_rows(train, test_eval) 
} else {
  full_data <- train
}

full_data$win <- factor(full_data$win, levels = c(0, 1))

rf_full_all <- fit(wf_rf, data = full_data)
# Extract variable importance again:
rf_all_fit <- extract_fit_parsnip(rf_full_all)$fit
vi_all <- enframe(rf_all_fit$variable.importance, name = "feature", value = "importance") %>%
  arrange(desc(importance))
write_csv(vi_all, out_rf_vi_full_csv)
message("Wrote final RF variable importance to: ", out_rf_vi_full_csv)
ggsave(out_rf_vi_full_png, vip(rf_all_fit, num_features = min(12, length(feature_cols))), width = 7, height = 5, dpi = 150)
message("Wrote final RF variable importance plot to: ", out_rf_vi_full_png)
