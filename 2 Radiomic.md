## ICC
```python
import pingouin as pg
import pandas as pd
import numpy as np
import os
from pandas import Series, DataFrame

# 读取表格A和表格B
excel_file_1 = "E:\A-50.xlsx"
excel_file_2 = "E:\B-50.xlsx"

data_1 = pd.read_excel(excel_file_1)
data_2 = pd.read_excel(excel_file_2)

data_1.insert(0, 'reader', np.ones(data_1.shape[0]))
data_2.insert(0, 'reader', np.ones(data_2.shape[0])*2)
data_1.insert(0, 'target', range(data_1.shape[0]))
data_2.insert(0, 'target', range(data_2.shape[0]))

data = pd.concat([data_1, data_2])
data.insert(0, 'new_column', data.iloc[:, 0])
data.to_excel(r'E:/Inter-ICC50.xlsx', index=False)

rows = data.shape[0]
cols = data.shape[1]
target = data.iloc[:, [1]]
reader = data.iloc[:, [2]]

ICC = np.zeros((6, cols-3))
for i in range(4, cols):
    feature = data.iloc[:, [i]]
    data0 = np.hstack((target, reader, feature))
    data0 = DataFrame(data0, columns = ["target", "reader", "feature"])
    icc = pg.intraclass_corr(data = data0, targets = 'target', raters = 'reader', ratings = 'feature')
    ICC[:, [i-3]] = icc.iloc[:, [2]]
ICC1 = DataFrame(ICC)
ICC1.to_excel(r'E:/After Inter-ICC50.xlsx', index=False)
```

```python
import pingouin as pg
import pandas as pd
import numpy as np
import os
from pandas import Series, DataFrame

# 读取表格A和表格B
excel_file_1 = "E:\A-30.xlsx"
excel_file_2 = "E:\B-30.xlsx"

data_1 = pd.read_excel(excel_file_1)
data_2 = pd.read_excel(excel_file_2)

data_1.insert(0, 'reader', np.ones(data_1.shape[0]))
data_2.insert(0, 'reader', np.ones(data_2.shape[0])*2)
data_1.insert(0, 'target', range(data_1.shape[0]))
data_2.insert(0, 'target', range(data_2.shape[0]))

data = pd.concat([data_1, data_2])
data.insert(0, 'new_column', data.iloc[:, 0])
data.to_excel(r'E:/Inter-ICC30.xlsx', index=False)

rows = data.shape[0]
cols = data.shape[1]
target = data.iloc[:, [1]]
reader = data.iloc[:, [2]]

ICC = np.zeros((6, cols-3))
for i in range(4, cols):
    feature = data.iloc[:, [i]]
    data0 = np.hstack((target, reader, feature))
    data0 = DataFrame(data0, columns = ["target", "reader", "feature"])
    icc = pg.intraclass_corr(data = data0, targets = 'target', raters = 'reader', ratings = 'feature')
    ICC[:, [i-3]] = icc.iloc[:, [2]]
ICC1 = DataFrame(ICC)
ICC1.to_excel(r'E:/After Inter-ICC30.xlsx', index=False)
```

## Person
```r
# Read the Excel file
library(readxl)
data <- read_excel("E:/A.xlsx")

# Calculate the Pearson correlation coefficient matrix
correlation_matrix <- cor(data)
write.csv(correlation_matrix, file = "E:/B.csv", row.names = FALSE)

# Find feature pairs with correlation coefficients greater than 0.8
high_correlation_pairs <- which(abs(correlation_matrix) > 0.8 & abs(correlation_matrix) < 1, arr.ind = TRUE)

# Remove redundant features
redundant_features <- character(0)
for (i in 1:nrow(high_correlation_pairs)) {
  row_index <- high_correlation_pairs[i, "row"]
  col_index <- high_correlation_pairs[i, "col"]
  feature1 <- rownames(correlation_matrix)[row_index]
  feature2 <- colnames(correlation_matrix)[col_index]

  if (!(feature1 %in% redundant_features) & !(feature2 %in% redundant_features)) {
    mean_correlation_feature1 <- mean(correlation_matrix[feature1, -which(colnames(correlation_matrix) == feature1)])
    mean_correlation_feature2 <- mean(correlation_matrix[feature2, -which(colnames(correlation_matrix) == feature2)])

    if (mean_correlation_feature1 > mean_correlation_feature2) {
      redundant_features <- c(redundant_features, feature2)
    } else {
      redundant_features <- c(redundant_features, feature1)
    }
  }
}

# Remove redundant features
data_filtered <- data[, !colnames(data) %in% redundant_features]

# Save the result as a CSV file
write.csv(data_filtered, "E:/C.csv", row.names = FALSE)

# Process the validation set
data2 <- read_excel("E:/D.xlsx")
data2_filtered <- data2[, !colnames(data2) %in% redundant_features]
write.csv(data2_filtered, file = "E:/E.csv", row.names = FALSE)
```

## PyRadiomic
```python
# This is an example of a parameters file
# It is written according to the YAML-convention (www.yaml.org) and is checked by the code for consistency.
# Three types of parameters are possible and reflected in the structure of the document:
#
# Parameter category:
#   Setting Name: <value>
#
# The three parameter categories are:
# - setting: Setting to use for preprocessing and class specific settings. if no <value> is specified, the value for
#   this setting is set to None.
# - featureClass: Feature class to enable, <value> is list of strings representing enabled features. If no <value> is
#   specified or <value> is an empty list ('[]'), all features for this class are enabled.
# - imageType: image types to calculate features on. <value> is custom kwarg settings (dictionary). if <value> is an
#   empty dictionary ('{}'), no custom settings are added for this input image.
#
# Some parameters have a limited list of possible values. Where this is the case, possible values are listed in the
# package documentation
 
# Settings to use, possible settings are listed in the documentation (section "Customizing the extraction").
setting:
  binWidth: 5    # Set 1,2,3,4,5,6,7,8,9,10
  label: 1
  interpolator: 'sitkBSpline' # This is an enumerated value, here None is not allowed
  resampledPixelSpacing: [1, 1, 1]# This disables resampling, as it is interpreted as None, to enable it, specify spacing in x, y, z as [x, y , z]
  weightingNorm: # If no value is specified, it is interpreted as None
  geometryTolerance: 0.0001
  normalize: False
 
# Image types to use: "Original" for unfiltered image, for possible filters, see documentation.
imageType:
  Original: {}
  Square: {}
  SquareRoot: {}
  Logarithm: {}
  Exponential: {}
  LoG: # If the in-plane spacing is large (> 2mm), consider removing sigma value 1.
    sigma: [0.5, 1.0, 1.5, 2.0]
  Wavelet:
    wavelet: 'rbio1.1'
    binWidth: 10
  LBP3D:
    lbp3DLevels: 2
    lbp3DIcosphereRadius: 1
    lbp3DIcosphereSubdivision: 1
  Gradient: {}
# Featureclasses, from which features must be calculated. If a featureclass is not mentioned, no features are calculated
# for that class. Otherwise, the specified features are calculated, or, if none are specified, all are calculated.

featureClass:
  shape:  # disable redundant Compactness 1 and Compactness 2 features by specifying all other shape features
  firstorder: 
  glcm:  
    - 'Autocorrelation'
    - 'ClusterProminence'
    - 'ClusterShade'
    - 'ClusterTendency'
    - 'Contrast'
    - 'Correlation'
    - 'DifferenceAverage'
    - 'DifferenceEntropy'
    - 'DifferenceVariance'
    - 'Id'
    - 'Idm'
    - 'Idmn'
    - 'Idn'
    - 'Imc1'
    - 'Imc2'
    - 'InverseVariance'
    - 'JointAverage'
    - 'JointEnergy'
    - 'JointEntropy'
    - 'MCC'
    - 'MaximumProbability'
    - 'SumEntropy'
    - 'SumSquares'
    - 'SumAverage'
  glrlm: # for lists none values are allowed, in this case, all features are enabled
  glszm:
  ngtdm:
  gldm:
```

```python
import radiomics
from radiomics import featureextractor
import pandas as pd
import scipy
import trimesh
import os
import time
import SimpleITK as sitk
import nibabel as nib
import numpy as np

dataDir = 'H:/A/'
extractor = featureextractor.RadiomicsFeatureExtractor() 
df = pd.DataFrame()
para_name = r'E:/exampleCT.yaml'

# Tissue
start_time = time.time()
for folder in os.listdir(dataDir):
        ori_path = dataDir + folder + '/data.nii.gz' 
        lab_path = dataDir + folder + '/mask.nii.gz'
        para_path = para_name     
        extractor = featureextractor.RadiomicsFeatureExtractor(para_path,label = 1)
        result = extractor.execute(ori_path,lab_path)        
        df_add = pd.DataFrame.from_dict(result.values()).T
        df_add.columns = result.keys()
        df_add['Name'] = folder
        df = pd.concat([df,df_add]) 
        print(folder)
end_time = time.time()
total_time = end_time - start_time
print("Time:", total_time, "s")
df.to_excel('E:/A.xlsx')
```

## LASSO
```r
####################################################################################################################
### title: "Prognosis - Grouping"
### author: "Kzzhu"
####################################################################################################################
### 1. Environment Configuration ------------------------------------------------------------------
{
  suppressMessages({
    library(readxl)
    library(glmnet)
    library(ggpubr) 
    library(cowplot)
    library(data.table)
    library(reshape2)
    library(ggplot2)
    library(ROCR)
    library(pROC)
  })
  rm(list = ls())
  options(stringsAsFactors = F)  # Remove duplicate option
  work_dir <- "E:/Radiomics"
  setwd(work_dir)
}

### 2. Data Import ------------------------------------------------------------------
Train <- read_excel("E:/Train-val.xlsx", sheet = 1)
X_Rad <- data.matrix(Train[, 6:ncol(Train)])  # Feature matrix
Y_Rad <- as.factor(make.names(Train$Diagnosis))  # Binary outcome
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)


### 3. Define Parameter Grids for 10-fold 10-repeated CV ------------------------------------------------------
# Parameter combinations: alpha (0=L2, 1=L1, 0.5=elastic net) + lambda sequence settings
param_grid <- list(
  # Model 1: Pure LASSO (L1 regularization)
  lasso = list(alpha = 1, lambda = NULL),
  # Model 2: Elastic Net (balanced L1+L2)
  elastic_net = list(alpha = 0.5, lambda = NULL),
  # Model 3: LASSO with custom lambda range (broader search)
  lasso_custom_lambda = list(alpha = 1, lambda = exp(seq(log(0.001), log(10), length.out = 200)))
)


### 4. 10-fold 10-repeated Cross-Validation ------------------------------------------------------
set.seed(123)
cv_results <- list()

# Perform 10 repetitions of 10-fold CV for each parameter set
for (model_name in names(param_grid)) {
  params <- param_grid[[model_name]]
  auc_values <- c()
  lambda_mins <- c()
  
  # 10 repetitions
  for (rep in 1:10) {
    # 10-fold CV for current repetition
    cv_fit <- cv.glmnet(
      x = X_Rad,
      y = Y_Rad,
      family = "binomial",
      alpha = params$alpha,
      lambda = params$lambda,
      nfolds = 10,  # 10-fold per repetition
      type.measure = "auc",
      foldid = sample(1:10, size = nrow(X_Rad), replace = TRUE)  # Different fold splits per repetition
    )
    
    # Store results
    auc_values[rep] <- max(cv_fit$cvm)  # Best AUC for this repetition
    lambda_mins[rep] <- cv_fit$lambda.min  # Optimal lambda for this repetition
  }
  
  # Summarize 10 repetitions
  cv_results[[model_name]] <- list(
    mean_auc = mean(auc_values),
    sd_auc = sd(auc_values),
    best_lambda = median(lambda_mins),  # Use median lambda across repetitions for stability
    all_lambdas = lambda_mins,
    all_aucs = auc_values
  )
  
  cat("Model", model_name, "complete: Mean AUC =", round(mean(auc_values), 3), 
      "(SD =", round(sd(auc_values), 3), ")\n")
}


### 5. Select Best Model and Refit ------------------------------------------------------
# Choose model with highest mean AUC
best_model_name <- names(which.max(sapply(cv_results, function(x) x$mean_auc)))
best_params <- param_grid[[best_model_name]]
best_lambda <- cv_results[[best_model_name]]$best_lambda

# Refit final model with best parameters
fit_final <- glmnet(
  x = X_Rad,
  y = Y_Rad,
  family = "binomial",
  alpha = best_params$alpha,
  lambda = best_lambda
)

# Save models and results
save(cv_results, file = "E:/Radiomics/LASSO_CVResults_10x10fold.RData")
save(fit_final, file = "E:/Radiomics/LASSO_BestModel_10x10fold.RData")


### 6. Visualization for Best Model ------------------------------------------------------
# 6.1 Cross-validation curve (using last CV run for visualization)
final_cv <- cv.glmnet(
  x = X_Rad,
  y = Y_Rad,
  family = "binomial",
  alpha = best_params$alpha,
  lambda = best_params$lambda,
  nfolds = 10,
  type.measure = "auc"
)
pdf("E:/Radiomics/LASSO1_10x10fold.pdf", width = 8, height = 6)
plot(final_cv, main = paste("10x10-fold CV -", best_model_name))
abline(v = log(best_lambda), lty = 2, col = "red", lwd = 2)
dev.off()

# 6.2 Coefficient path
pdf("E:/Radiomics/LASSO2_10x10fold.pdf", width = 8, height = 6)
plot(fit_final, xvar = "lambda", label = TRUE, 
     main = paste("Coefficient Path -", best_model_name))
abline(v = log(best_lambda), lty = 2, col = "red", lwd = 2)
dev.off()

# 6.3 Feature weights (non-zero coefficients)
coef_mat <- coef(fit_final, s = best_lambda)
active_coef <- data.table(
  feature = rownames(coef_mat)[coef_mat != 0],
  coef = as.numeric(coef_mat[coef_mat != 0])
)
if (nrow(active_coef) > 1) {  # Skip if only intercept
  active_coef <- active_coef[-1, ]  # Remove intercept
  active_coef <- active_coef[order(coef)]
  
  p <- ggplot(active_coef, aes(x = feature, y = coef)) +
    geom_bar(stat = "identity", fill = "#2258AA") +
    coord_flip() +
    labs(x = "Feature", y = "Coefficient", 
         title = paste("Active Features -", best_model_name)) +
    theme_classic()
  ggsave("E:/Radiomics/LASSO3_10x10fold.pdf", plot = p, width = 8, height = 6)
}


### 7. ROC Curve and Cutoff ------------------------------------------------------
# Get predicted probabilities (on logit scale for consistency with original code)
Y_predict <- predict(fit_final, newx = X_Rad, s = best_lambda, type = "link")
KZ <- cbind(Train$Diagnosis, Y_predict)

# Plot ROC
pdf("E:/Radiomics/LASSO-Train_10x10fold.pdf", width = 10, height = 10)
Train_roc <- plot.roc(
  KZ[, 1], KZ[, 2],
  main = paste("LASSO-Train (", best_model_name, ")", sep = ""),
  print.thres = "best",
  percent = TRUE, ci = TRUE, print.auc = TRUE
)
dev.off()

# Optimal cutoff (Youden's index)
cutoff <- coords(Train_roc, "best")[1, 1]
cat("Optimal cutoff for best model:", cutoff, "\n")


### 8. Output Results ------------------------------------------------------
output_df <- data.frame(
  Diagnosis = as.numeric(Train$Diagnosis),
  cutoff = cutoff,
  Y_predict = as.vector(Y_predict)
)
write.csv(output_df, file = "E:/Radiomics/Y_LASSOTrain_10x10fold.csv", row.names = FALSE)

# Save model comparison table
comparison_df <- data.frame(
  Model = names(cv_results),
  Alpha = sapply(param_grid, function(x) x$alpha),
  Mean_AUC = sapply(cv_results, function(x) x$mean_auc),
  SD_AUC = sapply(cv_results, function(x) x$sd_auc),
  Best_Lambda = sapply(cv_results, function(x) x$best_lambda)
)
write.csv(comparison_df, "E:/Radiomics/LASSO_ModelComparison_10x10fold.csv", row.names = FALSE)
```

```r
####################################################################################################################
### title: "Prognosis - Grouping"
### author: "Kzzhu"
####################################################################################################################
### 1. Environment Configuration ------------------------------------------------------------------
{
  suppressMessages({
    library(readxl)
    library(glmnet)
    library(ggpubr) 
    library(cowplot)
    library(data.table)
    library(reshape2)
    library(ggplot2)
    library(pROC)
  })
  rm(list = ls())
  options(stringsAsFactors = F)
  work_dir <- "E:/Radiomics"
  setwd(work_dir)
}

### 2. Data Import ------------------------------------------------------------------
load("E:/Radiomics/fitCV.RData")
load("E:/Radiomics/fitRad.RData")
Test <- read_excel("E:/Test.xlsx", sheet = 1)
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.factor(make.names(Test$Diagnosis)) 
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)

### 3. Plot ROC Curve for Test Set ---------------------------------------------------------
Y_predict <- predict(fitRad, X_Rad, s = fitCV$lambda.min, type = "link")
KZ <- cbind(Rad$Diagnosis, Y_predict)

pdf("E:/Radiomics/LASSO-Test.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "LASSO-Test",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
cutoff <- -0.4083606
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

### 4. Store Output Data Frame ------------------------------------------------------------
output_df <- data.frame(
  Diagnosis = as.numeric(Test$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Radiomics/Y LASSOTest.csv", row.names = F, col.names = T, sep = ",")

### 5. Plot ROC Curves for Internal and External Test Sets ---------------------------------------------------
Test <- read_excel("E:/Inter-Test.xlsx", sheet = 1)
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.factor(make.names(Test$Diagnosis)) 
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)
Y_predict <- predict(fitRad, X_Rad, s = fitCV$lambda.min, type = "link")
KZ <- cbind(Rad$Diagnosis, Y_predict)
pdf("E:/Radiomics/LASSO-InterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "LASSO-InterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

Test <- read_excel("E:/Exter-Test.xlsx", sheet = 1)
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.factor(make.names(Test$Diagnosis)) 
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)
Y_predict <- predict(fitRad, X_Rad, s = fitCV$lambda.min, type = "link")
KZ <- cbind(Rad$Diagnosis, Y_predict)
pdf("E:/Radiomics/LASSO-ExterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "LASSO-ExterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()
```

## SVM
```r
################################################################################
### title: "Prognosis - Grouping"
### author: "Kzzhu"
################################################################################
### 1. Environment Configuration-----------------------------------------------------------------
{
  suppressMessages({
    library(readxl)
    library(e1071)
    library(pROC)
    library(ggplot2)
    library(caret)
    library(kernlab)
  })
  rm(list = ls())
  options(stringsAsFactors = F)
  work_dir <- "E:/Radiomics"
  setwd(work_dir)
}

### 2. Data Import------------------------------------------------------------------
Train <- read_excel("E:/Train-val.xlsx", sheet = 1)
X_Rad <- data.matrix(Train[, 6:ncol(Train)])    # Feature data (radiomic features)
Y_Rad <- as.factor(make.names(Train$Diagnosis))  # Label data (binary outcome)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)      
dim(Rad)  # Check data dimension (samples x features + label)

### 3. Parameter Options for SVM Model (10-fold 10-repeated CV only)------------------------------------------------------
# 3.1 Cross-validation settings (only 10-fold 10-repeated)
cv_control <- trainControl(
  method = "repeatedcv", 
  number = 10,          # Fixed to 10-fold
  repeats = 10,         # 10 repetitions for robustness
  classProbs = TRUE,    # Enable class probabilities
  summaryFunction = twoClassSummary  # For binary classification metrics
)

# 3.2 SVM kernel & hyperparameter grids (multiple options)
# Option 1: Linear kernel with expanded C values
tune_grid_linear <- expand.grid(C = c(0.001, 0.0021, 0.01, 0.1, 1, 10))  # Regularization strength

# Option 2: RBF kernel (non-linear)
tune_grid_rbf <- expand.grid(
  C = c(0.1, 1, 10), 
  sigma = c(0.001, 0.01, 0.1)  # Kernel width (controls non-linearity)
)

# Option 3: Polynomial kernel (non-linear)
tune_grid_poly <- expand.grid(
  degree = c(2, 3),  # Polynomial degree
  scale = c(0.1, 1), 
  C = c(0.1, 1)      # Regularization strength
)

### 4. Train SVM Models with 10-fold 10-repeated CV------------------------------------------------------
set.seed(123)  # For reproducibility

# Model 1: Linear kernel + 10-fold 10-repeated CV
fitSVM_linear <- train(
  Diagnosis ~ ., 
  data = Rad, 
  method = "svmLinear", 
  trControl = cv_control, 
  metric = "ROC", 
  tuneGrid = tune_grid_linear
)
save(fitSVM_linear, file = "E:/Radiomics/fitSVM_linear.RData")

# Model 2: RBF kernel + 10-fold 10-repeated CV
fitSVM_rbf <- train(
  Diagnosis ~ ., 
  data = Rad, 
  method = "svmRadial", 
  trControl = cv_control, 
  metric = "ROC", 
  tuneGrid = tune_grid_rbf
)
save(fitSVM_rbf, file = "E:/Radiomics/fitSVM_rbf.RData")

# Model 3: Polynomial kernel + 10-fold 10-repeated CV
fitSVM_poly <- train(
  Diagnosis ~ ., 
  data = Rad, 
  method = "svmPoly", 
  trControl = cv_control, 
  metric = "ROC", 
  tuneGrid = tune_grid_poly
)
save(fitSVM_poly, file = "E:/Radiomics/fitSVM_poly.RData")

### 5. Evaluate Best Model (10-fold 10-repeated)------------------------------------------------------
# Compare models and select the best (based on CV ROC)
model_list <- list(
  linear = fitSVM_linear,
  rbf = fitSVM_rbf,
  poly = fitSVM_poly
)
cv_auc <- sapply(model_list, function(x) max(x$results$ROC))  # Extract best AUC for each model
best_model_name <- names(which.max(cv_auc))
best_model <- model_list[[best_model_name]]
cat("Best model:", best_model_name, "with CV AUC:", round(max(cv_auc), 3), "\n")

# Get predicted probabilities from best model
Y_predict <- predict(best_model, Rad, type = "prob")[, 2]
KZ <- cbind(actual = Rad$Diagnosis, predicted_prob = Y_predict)

# Plot ROC curve
pdf(paste0("E:/Radiomics/SVM_Best_", best_model_name, "_10fold10rep.pdf"), width = 10, height = 10)
Train_roc <- plot.roc(
  response = KZ[, "actual"], 
  predictor = KZ[, "predicted_prob"],
  main = paste("SVM (", best_model_name, ") - 10-fold 10-repeated CV", sep = ""),
  print.thres = "best",
  percent = TRUE,
  ci = TRUE,
  print.auc = TRUE
)
dev.off()

# Extract optimal cutoff (Youden's index)
cutoff <- coords(Train_roc, "best")[1, 1]
cat("Optimal cutoff for best model:", cutoff, "\n")

### 6. Save Output------------------------------------------------------------------
output_df <- data.frame(
  Diagnosis_actual = as.numeric(Train$Diagnosis),
  optimal_cutoff = cutoff,
  predicted_prob = Y_predict
)
write.csv(output_df, paste0("E:/Radiomics/SVM_", best_model_name, "_10fold10rep_Output.csv"), row.names = FALSE)

# Save model comparison results
model_comparison <- data.frame(
  model = names(model_list),
  best_cv_auc = cv_auc,
  best_hyperparams = sapply(model_list, function(x) paste(names(x$bestTune), "=", x$bestTune, collapse = ", "))
)
write.csv(model_comparison, "E:/Radiomics/SVM_ModelComparison_10fold10rep.csv", row.names = FALSE)
```

```r
####################################################################################################################
### title: "Prognosis - Grouping"
### author: "Kzzhu"
####################################################################################################################
### 1. Environment Configuration------------------------------------------------------------------
{
  suppressMessages({
    library(readxl)
    library(glmnet)
    library(ggpubr) 
    library(cowplot)
    library(data.table)
    library(reshape2)
    library(ggplot2)
    library(pROC)
  })
  rm(list = ls())
  options(stringsAsFactors = F)
  work_dir <- "E:/Radiomics"
  setwd(work_dir)
}

### 2. Data Import------------------------------------------------------------------
load("E:/Radiomics/fitSVMCV.RData")
Test <- read_excel("E:/Test.xlsx", sheet = 1)
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.factor(Test$Diagnosis)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)

### 3. Plot ROC Curve---------------------------------------------------------------
Y_predict <- predict(fitSVMCV, Rad, type = "prob")[, 2]  # Get probabilities for the positive class
KZ <- cbind(Rad$Diagnosis, Y_predict)
pdf("E:/Radiomics/SVMCV-Test.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "SVM-Test",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
cutoff <- 0.4020429
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
# dev.off()

### 4. Save Output Data Frame------------------------------------------------------------
output_df <- data.frame(
  Diagnosis = as.numeric(Test$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Radiomics/Y SVMTest.csv", row.names = F, col.names = T, sep = ",")

### 5. Plot ROC Curves for Internal and External Test Sets---------------------------------------------------
# Internal Test Set
Test <- read_excel("E:/Inter-Test.xlsx", sheet = 1)
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.factor(Test$Diagnosis)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)
Y_predict <- predict(fitSVMCV, Rad, type = "prob")[, 2]  # Get probabilities for the positive class
KZ <- cbind(Rad$Diagnosis, Y_predict)
pdf("E:/Radiomics/SVMCV-InterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "SVM-Internal Test",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

# External Test Set
Test <- read_excel("E:/Exter-Test.xlsx", sheet = 1)
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.factor(Test$Diagnosis)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)
Y_predict <- predict(fitSVMCV, Rad, type = "prob")[, 2]  # Get probabilities for the positive class
KZ <- cbind(Rad$Diagnosis, Y_predict)
pdf("E:/Radiomics/SVMCV-ExterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "SVM-External Test",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()
```

## RF
```r
################################################################################
### title: "Prognosis - Grouping"
### author: "Kzzhu"
################################################################################
### 1. Environment Configuration-----------------------------------------------------------------
{
  suppressMessages({
    library(readxl)
    library(randomForest)
    library(pROC)
    library(ggplot2)
    library(caret)
  })
  rm(list = ls())
  options(stringsAsFactors = F)
  work_dir <- "E:/Radiomics"
  setwd(work_dir)
}

### 2. Read Data-----------------------------------------------------------------
Train <- read_excel("E:/Train-val.xlsx", sheet = 1)
X_Rad <- data.matrix(Train[, 6:ncol(Train)])
Y_Rad <- as.factor(make.names(Train$Diagnosis))   # Label data
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)

### 3. Define Cross-Validation and Hyperparameter Grids-----------------------------------------
# 3.1 Cross-validation settings (only 10-fold 10-repeated)
cv_control <- trainControl(
  method = "repeatedcv", 
  number = 10,          # 10 folds
  repeats = 10,         # 10 repetitions
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

# 3.2 Hyperparameter grids (multiple options)
# Option 1: Vary mtry (number of variables randomly sampled as candidates at each split)
tune_grid_1 <- expand.grid(mtry = c(10, 20, 30, 40, 50, 60))  # Test a range around p/3

# Option 2: Combine mtry with other parameters
tune_grid_2 <- expand.grid(
  mtry = c(20, 40, 60),
  ntree = c(1000, 1500, 2000),  # Number of trees (higher for stability)
  nodesize = c(5, 10, 15),      # Minimum size of terminal nodes
  maxnodes = c(20, 30, 40)      # Maximum number of terminal nodes
)

### 4. Train Random Forest Models with 10-fold 10-repeated CV--------------------------------------
set.seed(10)  # For reproducibility

# Model 1: Vary mtry
fitRF_mtry <- train(
  Diagnosis ~ ., 
  data = Rad, 
  method = "rf", 
  trControl = cv_control, 
  metric = "ROC",
  tuneGrid = tune_grid_1,
  ntree = 1000
)
save(fitRF_mtry, file = "E:/Radiomics/fitRF_mtry_10fold10rep.RData")

# Model 2: Comprehensive parameter search
fitRF_comprehensive <- train(
  Diagnosis ~ ., 
  data = Rad, 
  method = "rf", 
  trControl = cv_control, 
  metric = "ROC",
  tuneGrid = tune_grid_2
)
save(fitRF_comprehensive, file = "E:/Radiomics/fitRF_comprehensive_10fold10rep.RData")

### 5. Evaluate Best Model and Plot ROC----------------------------------------------------------
# Select the best model based on CV ROC
model_list <- list(
  mtry_model = fitRF_mtry,
  comprehensive_model = fitRF_comprehensive
)
cv_auc <- sapply(model_list, function(x) max(x$results$ROC))
best_model_name <- names(which.max(cv_auc))
best_model <- model_list[[best_model_name]]
cat("Best model:", best_model_name, "with CV ROC:", round(max(cv_auc), 4), "\n")
cat("Best hyperparameters:", paste(names(best_model$bestTune), best_model$bestTune, collapse = ", "), "\n")

# Get predicted probabilities
Y_predict <- predict(best_model, Rad, type = "prob")[, 2]
KZ <- cbind(Rad$Diagnosis, Y_predict)

# Plot ROC curve
pdf(paste0("E:/Radiomics/RF_", best_model_name, "_10fold10rep.pdf"), width = 10, height = 10)
Train_roc <- plot.roc(
  KZ[, 1], 
  KZ[, 2],
  main = paste("Random Forest (", best_model_name, ") - 10-fold 10-repeated CV", sep = ""),
  print.thres = "best",
  percent = TRUE,
  ci = TRUE,
  print.auc = TRUE
)
dev.off()

# Extract optimal cutoff
cutoff <- coords(Train_roc, "best")[1, 1]
cat("Optimal cutoff:", cutoff, "\n")

### 6. Save Output Data Frame------------------------------------------------------------------
output_df <- data.frame(
  Diagnosis = as.numeric(Train$Diagnosis),
  cutoff = rep(cutoff, nrow(Rad)),
  Y_Rpredict = Y_predict
)
write.csv(output_df, file = paste0("E:/Radiomics/Y_RF_", best_model_name, "_10fold10rep.csv"), row.names = FALSE)

### 7. Compare Model Performance---------------------------------------------------
# Create comparison table
comparison <- data.frame(
  Model = names(model_list),
  Best_ROC = sapply(model_list, function(x) max(x$results$ROC)),
  Best_mtry = sapply(model_list, function(x) x$bestTune$mtry),
  ntree = sapply(model_list, function(x) ifelse("ntree" %in% names(x$bestTune), x$bestTune$ntree, NA)),
  nodesize = sapply(model_list, function(x) ifelse("nodesize" %in% names(x$bestTune), x$bestTune$nodesize, NA)),
  maxnodes = sapply(model_list, function(x) ifelse("maxnodes" %in% names(x$bestTune), x$bestTune$maxnodes, NA))
)
print(comparison)
write.csv(comparison, "E:/Radiomics/RF_ModelComparison_10fold10rep.csv", row.names = FALSE)

# Plot hyperparameter effects for mtry model
if (best_model_name == "mtry_model") {
  ggplot(fitRF_mtry$results, aes(x = mtry, y = ROC)) +
    geom_line() +
    geom_point() +
    geom_errorbar(aes(ymin = ROC - ROCSD, ymax = ROC + ROCSD), width = 2) +
    labs(title = "Effect of mtry on ROC (10-fold 10-repeated CV)", x = "mtry", y = "ROC") +
    theme_minimal()
  ggsave("E:/Radiomics/RF_mtry_vs_ROC_10fold10rep.png", width = 8, height = 6)
}
```

```r
################################################################################
### title: "Prognosis - Grouping"
### author: "Kzzhu"
################################################################################
### 1. Environment Configuration-----------------------------------------------------------------
{
  suppressMessages({
    library(readxl)
    library(randomForest)
    library(pROC)
    library(ggplot2)
  })
  rm(list = ls())
  options(stringsAsFactors = F)
  work_dir <- "E:/Radiomics"
  setwd(work_dir)
}

### 2. Data Import------------------------------------------------------------------
load("E:/Radiomics/fitRFCV.RData")
Test <- read_excel("E:/Test.xlsx", sheet = 1)
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.factor(Test$Diagnosis)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)

### 3. Plot ROC Curve for Test Set---------------------------------------------------------
Y_predict <- predict(fitRFCV, Rad, type = "prob")[, 2]
KZ <- cbind(Rad$Diagnosis, Y_predict)  # Combine labels and predicted probabilities
pdf("E:/Radiomics/RFCV-Test.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2], 
                     main = "RFCV-Test", 
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
cutoff <- 0.4198039
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

### 4. Save Output Data Frame------------------------------------------------------------
output_df <- data.frame(
  Diagnosis = as.numeric(Test$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Radiomics/Y RFTest.csv", row.names = F, col.names = T, sep = ",")

### 5. Plot ROC Curves for Internal and External Test Sets---------------------------------------------------
Test <- read_excel("E:/Inter-Test.xlsx", sheet = 1)
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.factor(Test$Diagnosis)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)
Y_predict <- predict(fitRFCV, Rad, type = "prob")[, 2]
KZ <- cbind(Rad$Diagnosis, Y_predict)  # Combine labels and predicted probabilities
pdf("E:/Radiomics/RFCV-InterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2], 
                     main = "RFCV-InterTest", 
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

Test <- read_excel("E:/Exter-Test.xlsx", sheet = 1)
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.factor(Test$Diagnosis)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)
Y_predict <- predict(fitRFCV, Rad, type = "prob")[, 2]
KZ <- cbind(Rad$Diagnosis, Y_predict)  # Combine labels and predicted probabilities
pdf("E:/Radiomics/RFCV-ExterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2], 
                     main = "RFCV-ExterTest", 
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()
```

## XGBoost
```r
################################################################################
### title: "Prognosis - Grouping"
### author: "Kzzhu"
################################################################################
### 1. Environment Configuration-----------------------------------------------------------------
{
  suppressMessages({
    library(readxl)
    library(xgboost)
    library(pROC)
    library(ggplot2)
  })
  rm(list = ls())
  options(stringsAsFactors = F)
  work_dir <- "E:/Radiomics"
  setwd(work_dir)
}

### 2. Data Import------------------------------------------------------------------
load("E:/Radiomics/fitXGBCV.RData")
Test <- read.table("E:/Test.csv", header = TRUE, sep = ",")
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.numeric(Test$Diagnosis)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)

### 3. Plot ROC Curve for Test Set---------------------------------------------------------
Y_predict <- predict(fitXGBCV, Rad, type = "prob")[, 2]
KZ <- cbind(Rad$Diagnosis, Y_predict)  # Combine labels and predicted probabilities
pdf("E:/Radiomics/XGBCV-Test.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "XGBCV-Test",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
cutoff <- 0.4729861
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

### 4. Save Output Data Frame------------------------------------------------------------
output_df <- data.frame(
  Diagnosis = as.numeric(Test$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Radiomics/Y XGBTest.csv", row.names = F, col.names = T, sep = ",")

### 5. Plot ROC Curves for Internal and External Test Sets---------------------------------------------------
Test <- read.table("E:/Inter-Test.csv", header = TRUE, sep = ",")
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.numeric(Test$Diagnosis)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)
Y_predict <- predict(fitXGBCV, Rad, type = "prob")[, 2]
KZ <- cbind(Rad$Diagnosis, Y_predict)  # Combine labels and predicted probabilities
pdf("E:/Radiomics/XGBCV-InterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "XGBCV-InterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

Test <- read.table("E:/Exter-Test.csv", header = TRUE, sep = ",")
X_Rad <- data.matrix(Test[, 6:ncol(Test)])
Y_Rad <- as.numeric(Test$Diagnosis)
Rad <- data.frame(Diagnosis = Y_Rad, X_Rad)
dim(Rad)
Y_predict <- predict(fitXGBCV, Rad, type = "prob")[, 2]
KZ <- cbind(Rad$Diagnosis, Y_predict)  # Combine labels and predicted probabilities
pdf("E:/Radiomics/XGBCV-ExterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "XGBCV-ExterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()
```

