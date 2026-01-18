### Clinical
```r
# Load necessary packages
library(readxl)
library(pROC)
library(ggplot2)
library(gridExtra)

# Read data
file_path <- "E:/Train-1.xlsx"
data <- read_excel(file_path)

# Select variables
variables <- c('Diagnosis', 'Eosinophil', 'Eosinophil_percent', 'Surgery', 'AR', 'Asthma', 'Atopy')

# Perform stepwise regression (backward elimination)
formula <- as.formula(paste("Diagnosis ~", paste(variables[-1], collapse = " + ")))
fitClinical <- step(glm(formula, data = data, family = binomial), direction = "backward")
summary(fitClinical)
save(fitClinical, file = "E:/fitClinical.RData")

# Plot ROC curve for the stepwise-selected model
Y_predict <- predict(fitClinical, data, type = "response")
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/Clinical-Train.pdf", width = 10, height = 10)
Train_roc <- plot.roc(KZ[, 1], KZ[, 2],
                      main = "Clinical-Train",
                      print.thres = "best",
                      percent = TRUE, ci = TRUE, print.auc = TRUE)
dev.off()
cutoff <- coords(Train_roc, "best")[1, 1]
cutoff

# Save output data
output_df <- data.frame(
  Diagnosis = as.numeric(data$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Y ClinicalTrain.csv", row.names = FALSE, col.names = TRUE, sep = ",")
cutoff
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
  work_dir <- "E:"
  setwd(work_dir)
}

### 2. Data Import------------------------------------------------------------------
load("E:/fitClinical.RData")
data <- read_excel("E:/Test-1.xlsx", sheet = 1)

# Select variables
variables <- c('Diagnosis', 'Eosinophil', 'Eosinophil_percent')

### 3. Plot ROC Curve---------------------------------------------------------------
Y_predict <- predict(fitClinical, data, type = "response")  # Get probabilities for the positive class
KZ <- cbind(data$Diagnosis, Y_predict)

pdf("E:/Clinical-Test.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "Clinical-Test",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
cutoff <- 0.3217003
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

### 4. Save Output Data Frame------------------------------------------------------------
output_df <- data.frame(
  Diagnosis = as.numeric(data$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Y ClinicalTest.csv", row.names = F, col.names = T, sep = ",")

### 5. Plot ROC Curves for Internal and External Test Sets---------------------------------------------------
# Internal test set
data <- read_excel("E:/Inter-Test-1.xlsx", sheet = 1)
variables <- c('Diagnosis', 'Eosinophil', 'Eosinophil_percent')
Y_predict <- predict(fitClinical, data, type = "response")  # Get probabilities for the positive class
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/Clinical-InterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "Clinical-InterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

# External test set
data <- read_excel("E:/Exter-Test-1.xlsx", sheet = 1)
variables <- c('Diagnosis', 'Eosinophil', 'Eosinophil_percent')
Y_predict <- predict(fitClinical, data, type = "response")  # Get probabilities for the positive class
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/Clinical-ExterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "Clinical-ExterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()
```

## CT-score
```r
# Load necessary packages
library(readxl)
library(pROC)
library(ggplot2)
library(gridExtra)

# Read data
file_path <- "E:/Train-2.xlsx"
data <- read_excel(file_path)

# Select variables
variables <- c('Diagnosis', 'TOTAL', 'EM')

# Create an empty list to store ROC objects
roc_list <- list()

# Multiple stepwise regression (backward elimination)
formula <- as.formula(paste("Diagnosis ~", paste(variables[-1], collapse = " + ")))
fit <- step(glm(formula, data = data, family = binomial), direction = "backward")
summary(fit)
save(fit, file = "E:/fitCTscore.RData")

# ROC curve after multiple stepwise regression
Y_predict <- predict(fit, data, type = "response")
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/CTscore-Train.pdf", width = 10, height = 10)
Train_roc <- plot.roc(KZ[, 1], KZ[, 2],
                      main = "CTscore-Train",
                      print.thres = "best",
                      percent = TRUE, ci = TRUE, print.auc = TRUE)
dev.off()
cutoff <- coords(Train_roc, "best")[1, 1]
cutoff

# Save data
output_df <- data.frame(
  Diagnosis = as.numeric(data$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Y CTscoreTrain.csv", row.names = FALSE, col.names = TRUE, sep = ",")
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
  work_dir <- "E:"
  setwd(work_dir)
}

### 2. Data Import------------------------------------------------------------------
load("E:/fitCTscore.RData")
data <- read_excel("E:/Test-2.xlsx", sheet = 1)

# Select variables
variables <- c('Diagnosis', 'TOTAL', 'EM')

### 3. Plot ROC Curve---------------------------------------------------------------
Y_predict <- predict(fit, data, type = "response")  # Get probabilities for the positive class
KZ <- cbind(data$Diagnosis, Y_predict)

pdf("E:/CTscore-Test.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "CTscore-Test",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
cutoff <- 0.2879729
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

### 4. Save Output Data Frame------------------------------------------------------------
Difference <- Y_predict - cutoff
Predict_class <- ifelse(Difference < 0, 0, 1)
output_df <- data.frame(
  Diagnosis = as.numeric(data$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Y CTscoreTest.csv", row.names = F, col.names = T, sep = ",")

### 5. Plot ROC Curves for Internal and External Test Sets---------------------------------------------------
# Internal test set
data <- read_excel("E:/Inter-Test-2.xlsx", sheet = 1)
variables <- c('Diagnosis', 'TOTAL', 'EM')
Y_predict <- predict(fit, data, type = "response")  # Get probabilities for the positive class
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/CTscore-InterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "CTscore-InterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

# External test set
data <- read_excel("E:/Exter-Test-2.xlsx", sheet = 1)
variables <- c('Diagnosis', 'TOTAL', 'EM')
Y_predict <- predict(fit, data, type = "response")  # Get probabilities for the positive class
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/CTscore-ExterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "CTscore-ExterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()
```

## Combined
```r
# Load necessary packages
library(readxl)
library(pROC)
library(ggplot2)
library(gridExtra)

# Read data
file_path <- "E:/Train-3.xlsx"
data <- read_excel(file_path)

# Select variables
variables <- c('Diagnosis', 'Eosinophil', 'XGB')
variables <- c('Diagnosis', 'Eosinophil', 'ResNet')
variables <- c('Diagnosis', 'Eosinophil', 'TOTAL', 'EM')

# Create empty list to store ROC objects
roc_list <- list()

# Multiple stepwise regression - backward method
formula <- as.formula(paste("Diagnosis ~", paste(variables[-1], collapse = " + ")))
fit <- step(glm(formula, data = data, family = binomial), direction = "backward")
summary(fit)
save(fit, file = "E:/fitCombined.RData")

# ROC curve after multiple stepwise regression
Y_predict <- predict(fit, data, type = "response")
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/Combined-Train.pdf", width = 10, height = 10)
Train_roc <- plot.roc(KZ[, 1], KZ[, 2],
                      main = "Combined-Train",
                      print.thres = "best",
                      percent = T, ci = T, print.auc = T)
dev.off()
cutoff <- coords(Train_roc, "best")[1, 1]
cutoff

# Save data
output_df <- data.frame(
  Diagnosis = as.numeric(data$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Y CombinedTrain.csv", row.names = F, col.names = T, sep = ",")    
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
  work_dir <- "E:"
  setwd(work_dir)
}

### 2. Data Import ------------------------------------------------------------------
load("E:/fitCombined.RData")
file_path <- "E:/Test-3.xlsx"
data <- read_excel(file_path)

# Select variables
variables <- c('Diagnosis', 'Eosinophil', 'XGB')
variables <- c('Diagnosis', 'Eosinophil', 'ResNet')
variables <- c('Diagnosis', 'Eosinophil', 'TOTAL', 'EM')

### 3. Plot ROC Curve ---------------------------------------------------------------
Y_predict <- predict(fit, data, type = "response")  # Get probabilities for the positive class
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/Combined-Test.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "Combined-Test",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
cutoff <- 0.4237492
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
# dev.off()

### 4. Save Output Data Frame ------------------------------------------------------------
output_df <- data.frame(
  Diagnosis = as.numeric(data$Diagnosis),
  cutoff = cutoff,
  Y_predict = Y_predict)
write.table(output_df, "E:/Y CombinedTest.csv", row.names = F, col.names = T, sep = ",")

### 5. Plot ROC for Internal and External Test Sets ---------------------------------------------------
# Internal Test Set
data <- read_excel("E:/Inter-Test-3.xlsx", sheet = 1)
variables <- c('Diagnosis', 'Eosinophil', 'LASSO')
Y_predict <- predict(fit, data, type = "response")  # Get probabilities for the positive class
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/Combined-InterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "Combined-InterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()

# External Test Set
data <- read_excel("E:/Exter-Test-3.xlsx", sheet = 1)
variables <- c('Diagnosis', 'Eosinophil', 'LASSO')
Y_predict <- predict(fit, data, type = "response")  # Get probabilities for the positive class
KZ <- cbind(data$Diagnosis, Y_predict)
pdf("E:/Combined-ExterTest.pdf", width = 10, height = 10)
Test_roc <- plot.roc(KZ[, 1], KZ[, 2],
                     main = "Combined-ExterTest",
                     percent = TRUE, ci = TRUE, print.auc = TRUE)
sens <- coords(Test_roc, x = cutoff, input = "threshold")$sensitivity
spec <- coords(Test_roc, x = cutoff, input = "threshold")$specificity
points(spec, sens, col = "red", pch = 19, cex = 1.2)
text(spec, sens, labels = paste("Sens:", round(sens, 2), "\nSpec:", round(spec, 2)), pos = 4, col = "red", cex = 1.2)
dev.off()    
```

## ROC
```r
library(readxl)
library(pROC)
library(ggplot2)

# Read data
file_path <- "E:/Train-4.xlsx"
df <- read_excel(file_path)

# Extract Diagnosis and its feature columns
target_col <- "Diagnosis"
selected_features <- c("XGB", "ResNet", "Eosinophil", "CTscore", "Combined1", "Combined2", "Combined3")

data <- df[, c(target_col, selected_features)]
data$Diagnosis <- as.factor(data$Diagnosis)

# Color mapping
color_mapping <- c(
  "ML" = "#0073CF",          
  "DL" = "#008000",          
  "Eosinophil" = "#FFD700",  
  "CTscore" = "#F69100",     
  "Combined1" = "#F00B00",   
  "Combined2" = "#FF8787",   
  "Combined3" = "#9E1818"    
)

# Models that need Youden's points
youden_models <- c("XGB", "ResNet", "Eosinophil", "CTscore")

# Initialize lists
roc_results <- list()
auc_ci_values <- c()
youden_points <- list()

# Calculate ROC and AUC
for (feature in selected_features) {
  # Fit logistic regression model for each feature
  model <- glm(Diagnosis ~ ., data = data[, c(target_col, feature)], family = binomial)
  # Get predicted probabilities
  prob <- predict(model, type = "response")
  # Calculate ROC curve with confidence interval
  roc_curve <- roc(data$Diagnosis, prob, ci = TRUE)

  # Rename features for display (ML for XGB, DL for ResNet)
  new_name <- ifelse(feature == "XGB", "ML",
                     ifelse(feature == "ResNet", "DL", feature))

  # Store ROC results
  roc_results[[new_name]] <- roc_curve
  # Store AUC with 95% confidence interval
  auc_ci_values <- c(auc_ci_values, sprintf("%s: %.3f (%.3f - %.3f)", new_name,
                                            auc(roc_curve),
                                            ci.auc(roc_curve)[1],
                                            ci.auc(roc_curve)[3]))

  # Record Youden's points for specified models
  if (feature %in% youden_models) {
    best_coords <- coords(roc_curve, x = "best", best.method = "youden", ret = c("specificity", "sensitivity"))
    youden_points[[new_name]] <- best_coords
  }
}

# Output as PDF
pdf("E:/ROC_Curves—Train.pdf", width = 6, height = 6)

# Plot the first ROC curve
plot(roc_results[[1]], col = color_mapping[names(roc_results)[1]], 
     main = "ROC Curves with AUC (95% CI)", lwd = 2)

# Overlay subsequent ROC curves
for (i in 2:length(roc_results)) {
  plot(roc_results[[i]], col = color_mapping[names(roc_results)[i]], add = TRUE, lwd = 2)
}

# Add Youden's points (using triangles)
for (name in names(youden_points)) {
  coords <- youden_points[[name]]
  points(coords["specificity"], coords["sensitivity"],
         col = color_mapping[name], pch = 17, cex = 1.2)  # pch=17 for solid triangle
}

# Add legend with AUC values and 95% CI
legend("bottomright", legend = auc_ci_values, col = color_mapping[names(roc_results)], 
       lwd = 2, cex = 0.8)

# Close PDF device
dev.off()
```

```python
library(readxl)
library(pROC)
library(ggplot2)

# Read data
file_path <- "E:/Test-4.xlsx"
df <- read_excel(file_path)

# Extract variables
target_col <- "Diagnosis"
selected_features <- c("XGB", "ResNet", "Eosinophil", "CTscore", "Combined1", "Combined2", "Combined3")

# Color mapping
color_mapping <- c(
  "ML" = "#0073CF",        
  "DL" = "#008000",  
  "Eosinophil" = "#FFD700", 
  "CTscore" = "#F69100",    
  "Combined1" = "#F00B00", 
  "Combined2" = "#FF8787", 
  "Combined3" = "#9E1818" 
)

# Original column names of models requiring Youden's points annotation
youden_models <- c("XGB", "ResNet", "Eosinophil", "CTscore")

# Get all unique Cohort values
cohorts <- unique(df$Cohort)

# Open a PDF device for multiple pages
pdf("E:/All_ROC_Curves_by_Cohort.pdf", width = 6, height = 6)

# Loop through each Cohort
for (cohort in cohorts) {
  # Subset data for the current cohort
  data <- df[df$Cohort == cohort, c("Diagnosis", selected_features)]
  data$Diagnosis <- as.factor(data$Diagnosis)
  
  roc_results <- list()
  auc_ci_values <- c()
  youden_points <- list()
  
  for (feature in selected_features) {
    # Fit logistic regression model
    model <- glm(Diagnosis ~ ., data = data[, c(target_col, feature)], family = binomial)
    # Get predicted probabilities
    prob <- predict(model, type = "response")
    # Calculate ROC curve with confidence interval
    roc_curve <- roc(data$Diagnosis, prob, ci = TRUE)
    
    # Rename features for display
    new_name <- ifelse(feature == "XGB", "ML",
                       ifelse(feature == "ResNet", "DL", feature))
    
    roc_results[[new_name]] <- roc_curve
    
    # Store AUC with 95% confidence interval
    auc_ci_values <- c(auc_ci_values, sprintf("%s: %.3f (%.3f - %.3f)", new_name,
                                              auc(roc_curve),
                                              ci.auc(roc_curve)[1],
                                              ci.auc(roc_curve)[3]))
    
    # Record Youden's points for specified models
    if (feature %in% youden_models) {
      best_coords <- coords(roc_curve, x = "best", best.method = "youden", ret = c("specificity", "sensitivity"))
      youden_points[[new_name]] <- best_coords
    }
  }
  
  # Create a new plot page for each cohort
  plot(roc_results[[1]], col = color_mapping[names(roc_results)[1]],
       main = paste("ROC Curves -", cohort), lwd = 4, lty = 3)
  
  # Overlay other ROC curves
  for (i in 2:length(roc_results)) {
    plot(roc_results[[i]], col = color_mapping[names(roc_results)[i]], add = TRUE, lwd = 4, lty = 3)
  }
  
  # Add Youden's points (triangles)
  for (name in names(youden_points)) {
    coords <- youden_points[[name]]
    points(coords["specificity"], coords["sensitivity"],
           col = color_mapping[name], pch = 17, cex = 1.2)
  }
  
  # Add legend
  legend("bottomright", legend = auc_ci_values,
         col = color_mapping[names(roc_results)],
         lwd = 2, cex = 0.8, lty = 3)
}

# Close the PDF device
dev.off()
```

```r
library(readxl)
library(pROC)
library(ggplot2)

# Read data
file_path <-  "E:/Train.xlsx"
file_path <-  "E:/Inter-Test.xlsx"
file_path <-  "E:/Exter-Test.xlsx"
df <- read_excel(file_path)

# Target and feature columns
target_col <- "Diagnosis"
selected_features <- c ("XGB", #"LASSO", "SVM", "RF",
                       "ResNet", #"AlexNet", "VGG16", "DenseNet", "Vit3D", "Swin3D",
                       "Eosinophil", "CTscore", 
                       "Combined1", "Combined2", "Combined3")

# Extract data
data <- df[, c(target_col, selected_features, "Cohort")]
data$Diagnosis <- as.factor(data$Diagnosis)

# Get unique Cohort values (AA, BB, CC, DD, EE, FF, GG, HH)
cohorts <- unique(data$Cohort)

# Store ROC results and p-values for each cohort
roc_results <- list()
p_values <- c()

# Process each cohort
for (cohort in cohorts) {
  cohort_data <- subset(data, Cohort == cohort)  # Filter data for current cohort
  
  # Calculate ROC curve and AUC for roc1 (Combined1) and each roc2 feature
  roc1 <- roc(cohort_data$Diagnosis, cohort_data$Combined1)
  
  # Store p-values
  cohort_p_values <- c(paste0("P-value for ROC1 vs ROC2 in Cohort ", cohort))
  
  for (feature in selected_features) {
    if (feature != "Combined1") {  # Avoid comparing Combined1 with itself
      roc2 <- roc(cohort_data$Diagnosis, cohort_data[[feature]])
      delong_test <- roc.test(roc1, roc2, method = "delong")
      
      # Save p-value for this feature
      cohort_p_values <- c(cohort_p_values, 
                           paste0(feature, ": ", round(delong_test$p.value, 4)))
    }
  }
  
  # Save ROC results and p-values for each cohort
  roc_results[[cohort]] <- cohort_p_values
}

# Print results (p-values for each cohort)
for (cohort in cohorts) {
  cat(paste0("Cohort: ", cohort, "\n"))
  cat(paste(roc_results[[cohort]], collapse = "\n"), "\n\n")
}
```

## Calibration
```r
library(rms)
library(readxl)
library(ggplot2)
library(ResourceSelection)

# Read data
train <- read_excel("E:/Train.xlsx")
internal <- read_excel("E:/Inter-Test.xlsx")
external <- read_excel("E:/Exter-Test.xlsx")

# Convert data formats
datasets <- list(train = train, internal = internal, external = external)

for (name in names(datasets)) {
  datasets[[name]]$Diagnosis <- as.factor(datasets[[name]]$Diagnosis)
  datasets[[name]]$Combined1 <- as.numeric(datasets[[name]]$Combined1)
  datasets[[name]]$Combined2 <- as.numeric(datasets[[name]]$Combined2)
}

# Build models and calibration curves
calibrations <- list()
hl_tests <- list()

for (name in names(datasets)) {
  data <- datasets[[name]]
  # Build logistic regression models
  model1 <- lrm(Diagnosis ~ Combined1, data = data, x = TRUE, y = TRUE)
  model2 <- lrm(Diagnosis ~ Combined2, data = data, x = TRUE, y = TRUE)
  
  # Generate calibration curves with 1000 bootstrap samples
  calibrations[[paste0(name, "_1")]] <- calibrate(model1, B = 1000, data = data)
  calibrations[[paste0(name, "_2")]] <- calibrate(model2, B = 1000, data = data)
  
  # Perform Hosmer-Lemeshow test
  Y_Compredict <- predict(model1, data, type = "lp")
  Diagnosis <- as.numeric(data$Diagnosis)
  hl_tests[[name]] <- hoslem.test(Diagnosis, Y_Compredict, g = 10)  # Note: Fixed assignment to store results properly
}

# Plot calibration curves
pdf("E:/Calibration_Curves.pdf")
plot(1, type = "n", xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted Probability", ylab = "Observed Probability",
     main = "Calibration Curves for Combined Models")
abline(0, 1, col = "black", lty = 2)  # Reference line (perfect calibration)

# Define visual parameters
colors <- c("#F00B00", "#FF8787")
linetypes <- c("solid", "dashed", "dotted")
labels <- c("Train Combined1", "Train Combined2",
            "Internal Combined1", "Internal Combined2",
            "External Combined1", "External Combined2")

# Plot calibration curves
count <- 1
for (name in names(calibrations)) {
  lines(calibrations[[name]][, c("predy", "calibrated.corrected")],
        type = "l", lwd = 2, col = colors[(count %% 2) + 1], lty = linetypes[ceiling(count / 2)])
  count <- count + 1
}

# Add legend
legend("bottomright", legend = labels, col = rep(colors, each = 3),
       lty = rep(linetypes, each = 2), lwd = 2, bty = "o")
dev.off()

# Output Hosmer-Lemeshow test results
for (name in names(hl_tests)) {
  print(paste(name, "Hosmer-Lemeshow P-value:", hl_tests[[name]]$p.value))  # Note: Fixed to access list elements properly
}
```

## DCA
```r
library(readxl)
library(rmda)

# Read training set data
train <- read_excel("E:/Train.xlsx")

# Define color mapping
colors <- c("LASSO" = "#0073CF",
            "ResNet" = "#008000",
            "Eosinophil" = "#FFD700",
            "CTscore" = "#F69100",
            "Combined1" = "#F00B00",
            "Combined2" = "#FF8787",
            "Combined3" = "#9E1818")

# Calculate Decision Curve Analysis (DCA)
dca_lasso <- decision_curve(Diagnosis ~ LASSO, data = train, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_resnet <- decision_curve(Diagnosis ~ ResNet, data = train, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_eos <- decision_curve(Diagnosis ~ Eosinophil, data = train, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_ct <- decision_curve(Diagnosis ~ CTscore, data = train, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_comb1 <- decision_curve(Diagnosis ~ Combined1, data = train, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_comb2 <- decision_curve(Diagnosis ~ Combined2, data = train, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_comb3 <- decision_curve(Diagnosis ~ Combined3, data = train, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')

# Combine DCA curves
dca_list <- list(dca_lasso, dca_resnet, dca_eos, dca_ct, dca_comb1, dca_comb2, dca_comb3)

# Save to PDF
pdf("E:/DCA_Train.pdf", width = 8, height = 8)

# Plot DCA curves
plot_decision_curve(dca_list,
                    curve.names = names(colors),
                    lwd = 2.5,
                    col = colors,
                    lty = 1,  # Solid line
                    cost.benefit.axis = FALSE,
                    confidence.intervals = FALSE,
                    standardize = FALSE)

dev.off()  # Close PDF device

###-----------------------------------------------------------------------------
# Read internal test set data
test <- read_excel("E:/Inter-Test.xlsx")

# Calculate Decision Curve Analysis (DCA)
dca_lasso <- decision_curve(Diagnosis ~ LASSO, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_resnet <- decision_curve(Diagnosis ~ ResNet, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_eos <- decision_curve(Diagnosis ~ Eosinophil, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_ct <- decision_curve(Diagnosis ~ CTscore, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_comb1 <- decision_curve(Diagnosis ~ Combined1, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_comb2 <- decision_curve(Diagnosis ~ Combined2, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_comb3 <- decision_curve(Diagnosis ~ Combined3, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')

# Combine DCA curves
dca_list <- list(dca_lasso, dca_resnet, dca_eos, dca_ct, dca_comb1, dca_comb2, dca_comb3)

# Save to PDF
pdf("E:/DCA_InterTest.pdf", width = 8, height = 8)

# Plot DCA curves
plot_decision_curve(dca_list,
                    curve.names = names(colors),
                    lwd = 2.5,
                    col = colors,
                    lty = 1,  # Solid line
                    cost.benefit.axis = FALSE,
                    confidence.intervals = FALSE,
                    standardize = FALSE)

dev.off()  # Close PDF device

###-----------------------------------------------------------------------------
# Read external test set data
test <- read_excel("E:/Exter-Test.xlsx")

# Calculate Decision Curve Analysis (DCA)
dca_lasso <- decision_curve(Diagnosis ~ LASSO, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_resnet <- decision_curve(Diagnosis ~ ResNet, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_eos <- decision_curve(Diagnosis ~ Eosinophil, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_ct <- decision_curve(Diagnosis ~ CTscore, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_comb1 <- decision_curve(Diagnosis ~ Combined1, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_comb2 <- decision_curve(Diagnosis ~ Combined2, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')
dca_comb3 <- decision_curve(Diagnosis ~ Combined3, data = test, family = binomial(link = 'logit'), thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95, study.design = 'cohort')

# Combine DCA curves
dca_list <- list(dca_lasso, dca_resnet, dca_eos, dca_ct, dca_comb1, dca_comb2, dca_comb3)

# Save to PDF
pdf("E:/DCA_ExterTest.pdf", width = 8, height = 8)

# Plot DCA curves
plot_decision_curve(dca_list,
                    curve.names = names(colors),
                    lwd = 2.5,
                    col = colors,
                    lty = 1,  # Solid line
                    cost.benefit.axis = FALSE,
                    confidence.intervals = FALSE,
                    standardize = FALSE)

dev.off()  # Close PDF device    
```

## NRI and IDI
```r
# Load necessary libraries
library(readxl)
library(pROC)
library(caret)
library(rmda)
library(e1071)
library(PredictABEL)

# Read data (can be replaced with internal/external test sets)
file_path <- ("E:/Train.xlsx")
file_path <- ("E:/Inter-Test.xlsx")
file_path <- ("E:/Exter-Test.xlsx")
data <- read_excel(file_path)[, c("Diagnosis", "XGB", "ResNet", "Eosinophil", "CTscore")]

# Confirm required column names
needed_cols <- c("Diagnosis", "XGB", "ResNet", "Eosinophil", "CTscore")
data <- data[, needed_cols]

# Convert to data.frame to avoid compatibility issues with tibble
data <- as.data.frame(data)

# Ensure Diagnosis is numeric (0/1)
data$Diagnosis <- as.numeric(as.character(data$Diagnosis))

# Define function to calculate NRI and IDI
calculate_nri_idi <- function(baseline_var, newmodel_var, data, outcome_col = "Diagnosis") {
  # Select variables and outcome, remove missing values
  used_vars <- c(outcome_col, baseline_var, newmodel_var)
  data_clean <- na.omit(data[, used_vars])
  
  # Construct formulas
  f1 <- as.formula(paste(outcome_col, "~", baseline_var))
  f2 <- as.formula(paste(outcome_col, "~", newmodel_var))
  
  # Fit models
  model1 <- glm(f1, data = data_clean, family = binomial)
  model2 <- glm(f2, data = data_clean, family = binomial)
  
  # Predict probabilities
  pred1 <- predict(model1, type = "response")
  pred2 <- predict(model2, type = "response")
  
  # Calculate NRI/IDI
  nri_idi_result <- reclassification(data = data_clean,
                                     cOutcome = which(colnames(data_clean) == outcome_col),
                                     predrisk1 = pred1,
                                     predrisk2 = pred2,
                                     cutoff = seq(0, 1, by = 0.1))
  return(nri_idi_result)
}

# Run 4 groups of comparisons
res_XGB_vs_CTscore <- calculate_nri_idi("CTscore", "XGB", data)
res_XGB_vs_Eosinophil <- calculate_nri_idi("Eosinophil", "XGB", data)
res_ResNet_vs_CTscore <- calculate_nri_idi("CTscore", "ResNet", data)
res_ResNet_vs_Eosinophil <- calculate_nri_idi("Eosinophil", "ResNet", data)

# Print results
print("XGB vs Eosinophil")
print(res_XGB_vs_Eosinophil)

print("ResNet vs Eosinophil")
print(res_ResNet_vs_Eosinophil)

print("XGB vs CTscore")
print(res_XGB_vs_CTscore)

print("ResNet vs CTscore")
print(res_ResNet_vs_CTscore)
```

## Heatmap
```r
library(corrplot)
library(readxl)
library(pheatmap)
library(ggplot2)
library(ggtree)
library(tidyr)
library(aplot)
library(Hmisc)

rm(list = ls())
options(stringsAsFactors = F)

# Read data
data <- read_excel("E:/heatmap.xlsx")

# Extract specified columns
cols <- c("F", "E", "M", "S", "OMC", "TOTAL", "EM", "Age", "Leukocyte", "Neutrophil", 
          "Neutrophil_percent", "Lymphocyte", "Lymphocyte_percent", "Eosinophil", 
          "Eosinophil_percent", "Basophil", "Basophil_percent", "Monocyte", "Monocyte_percent",
          "LASSO", "SVM", "RF", "XGB", "AlexNet", "VGG16", "ResNet", 
          "DenseNet", "Vit3D", "Swin3D", 
          "Clinical","CTscore",
          "Combined1", "Combined2", "Combined3")
heatmap_data <- data[, cols]

# Calculate correlation matrix
mydata_cor <- cor(heatmap_data, method = "spearman")

# Keep all rows, retain only specified items in columns
selected_cols <- c("LASSO", "SVM", "RF", "XGB", "Combined1", 
                   "AlexNet", "VGG16", "ResNet", 
                   "DenseNet", "Vit3D", "Swin3D", "Combined2", 
                   "Clinical","CTscore", "Combined3")
selected_rows <- c("F", "E", "M", "S", "OMC", "TOTAL", "EM", "Age", "Leukocyte", "Neutrophil", 
                   "Neutrophil_percent", "Lymphocyte", "Lymphocyte_percent", "Eosinophil", 
                   "Eosinophil_percent", "Basophil", "Basophil_percent", "Monocyte", "Monocyte_percent")
cor_matrix <- mydata_cor[selected_rows, selected_cols]

# Calculate p-value matrix
mydata_cor2 <- rcorr(as.matrix(heatmap_data), type = "spearman")
pvalue <- mydata_cor2$P[selected_rows, selected_cols]

# Mark p-value significance
data_mark = matrix("", nrow = nrow(pvalue), ncol = ncol(pvalue))
for(i in 1:nrow(pvalue)){    
  for(j in 1:ncol(pvalue)){        
    if(pvalue[i,j] <= 0.001){data_mark[i,j] = "***"}            
    else if(pvalue[i,j] <= 0.01){data_mark[i,j] = "**"} 
    else if(pvalue[i,j] <= 0.05){data_mark[i,j] = "*"}            
    else {data_mark[i,j] = " "}}}

# Draw pheatmap
pheatmap(cor_matrix,
         cluster_rows = T,
         cluster_cols = F,
         border_color = "#7C858D",
         display_numbers = data_mark, 
         color = colorRampPalette(c("#0F3BA9", "white", "#FF0A0A"))(100))
dev.off()

# Convert to data frame
p <- data.frame(cor_matrix)
p$Feature <- rownames(p)

# Wide format to long format
p1 <- gather(p, key = "Model", value = 'correlation', -Feature)

# Draw ggplot heatmap and save to PDF
pdf("E:/heatmap.pdf", width = 6, height = 10) 
pheatmap(cor_matrix,
         cluster_rows = T,
         cluster_cols = F,
         border_color = "#7C858D",
         display_numbers = data_mark, 
         color = colorRampPalette(c("blue", "white", "red"))(100))
dev.off()  # Close PDF device
```



