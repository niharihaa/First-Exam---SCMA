# Install and load required libraries
install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

packages <- c("tidyverse", "caret", "car", "broom", "ggplot2", "lmtest", "MASS", "glmnet")
install_and_load(packages)

# Load the dataset
file_path <- "C:/Users/nihar/OneDrive/Desktop/Bootcamp/SCMA 632/DataSet/cancer_reg.csv"
cancer_data <- read.csv(file_path)

# Handle missing values
cancer_data <- cancer_data %>%
  mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .))

# Select relevant features
selected_features <- c("avgAnnCount", "avgDeathsPerYear", "incidenceRate", "medIncome", 
                       "popEst2015", "povertyPercent", "MedianAgeMale", "PercentMarried", 
                       "PctNoHS18_24", "PctHS18_24", "PctBachDeg18_24", "PctHS25_Over", 
                       "PctBachDeg25_Over", "PctEmployed16_Over", "PctPrivateCoverage", 
                       "PctEmpPrivCoverage", "PctWhite", "PctOtherRace", "PctMarriedHouseholds", 
                       "BirthRate", "PctPublicCoverage", "TARGET_deathRate")

# Check if selected features are present in the data
missing_features <- setdiff(selected_features, names(cancer_data))
if (length(missing_features) > 0) {
  stop(paste("Missing features in the dataset:", paste(missing_features, collapse = ", ")))
}

# Select the columns manually
cancer_data <- cancer_data[, selected_features]

# Identify outliers using Cook's distance on the entire dataset
full_model <- lm(TARGET_deathRate ~ ., data = cancer_data)
cooksd <- cooks.distance(full_model)

# Plot Cook's distance
plot(cooksd, type = "h", main = "Cook's Distance", ylab = "Cook's Distance")
abline(h = 4 / nrow(cancer_data), col = "red")

# Identify high influence points
influential_threshold <- 4 / nrow(cancer_data)
influential_points <- which(cooksd > influential_threshold)
print(influential_points)

# Cap the influential points at the threshold
cancer_data <- cancer_data %>%
  mutate(cooksd = cooksd) %>%
  mutate(cooksd = ifelse(cooksd > influential_threshold, influential_threshold, cooksd))

# Drop cooksd column after capping using base R
cancer_data <- cancer_data[, !names(cancer_data) %in% "cooksd"]

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(cancer_data$TARGET_deathRate, p = 0.8, list = FALSE)
train_data <- cancer_data[trainIndex, ]
test_data <- cancer_data[-trainIndex, ]

# Add more polynomial terms and interaction terms
train_data <- train_data %>%
  mutate(
    incidenceRate2 = incidenceRate^2,
    medIncome2 = medIncome^2,
    incidenceRate_medIncome = incidenceRate * medIncome,
    popEst2015_povertyPercent = popEst2015 * povertyPercent,
    PctHS18_24_PctBachDeg18_24 = PctHS18_24 * PctBachDeg18_24,
    PctPrivateCoverage_PctPublicCoverage = PctPrivateCoverage * PctPublicCoverage,
    povertyPercent2 = povertyPercent^2,
    PercentMarried_PctHS25_Over = PercentMarried * PctHS25_Over
  )

test_data <- test_data %>%
  mutate(
    incidenceRate2 = incidenceRate^2,
    medIncome2 = medIncome^2,
    incidenceRate_medIncome = incidenceRate * medIncome,
    popEst2015_povertyPercent = popEst2015 * povertyPercent,
    PctHS18_24_PctBachDeg18_24 = PctHS18_24 * PctBachDeg18_24,
    PctPrivateCoverage_PctPublicCoverage = PctPrivateCoverage * PctPublicCoverage,
    povertyPercent2 = povertyPercent^2,
    PercentMarried_PctHS25_Over = PercentMarried * PctHS25_Over
  )

# Fit the enhanced model with additional terms
enhanced_model_v2 <- lm(TARGET_deathRate ~ avgAnnCount + avgDeathsPerYear + incidenceRate + 
                          medIncome + popEst2015 + povertyPercent + MedianAgeMale + 
                          PercentMarried + PctNoHS18_24 + PctHS18_24 + PctBachDeg18_24 + 
                          PctHS25_Over + PctBachDeg25_Over + PctEmployed16_Over + 
                          PctPrivateCoverage + PctEmpPrivCoverage + PctWhite + PctOtherRace + 
                          PctMarriedHouseholds + BirthRate + incidenceRate2 + medIncome2 + 
                          incidenceRate_medIncome + popEst2015_povertyPercent + 
                          PctHS18_24_PctBachDeg18_24 + PctPrivateCoverage_PctPublicCoverage + 
                          povertyPercent2 + PercentMarried_PctHS25_Over, 
                        data = train_data)
summary(enhanced_model_v2)

# Model diagnostics
# Linearity
ggplot(data = train_data, aes(x = predict(enhanced_model_v2), y = TARGET_deathRate)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "blue") +
  labs(title = "Predicted vs Observed Values", x = "Predicted Values", y = "Observed Values")

# Residuals vs Fitted Values
ggplot(data = train_data, aes(x = predict(enhanced_model_v2), y = resid(enhanced_model_v2))) +
  geom_point() +
  geom_hline(yintercept = 0, color = "green") +
  labs(title = "Residuals vs Fitted Values", x = "Fitted Values", y = "Residuals")
  
# Q-Q plot for residuals
qqnorm(resid(enhanced_model_v2))
qqline(resid(enhanced_model_v2), col = "purple")
labs(title = "Residuals vs Fitted Values", x = "Fitted Values", y = "Residuals")

# Histogram of residuals
hist(resid(enhanced_model_v2), breaks = 30, main = "Histogram of Residuals", xlab = "Residuals")

# Serial independence of errors
dwtest(enhanced_model_v2)

# Heteroskedasticity
bptest(enhanced_model_v2)

# Normality of residuals
shapiro.test(resid(enhanced_model_v2))

# Multicollinearity
vif(enhanced_model_v2)

# Model evaluation on test data
test_predictions_v2 <- predict(enhanced_model_v2, newdata = test_data)

# Calculate R-squared
rsq <- function(actual, predicted) {
  cor(actual, predicted) ^ 2
}
test_rsq_v2 <- rsq(test_data$TARGET_deathRate, test_predictions_v2)

# Calculate RMSE
test_rmse_v2 <- sqrt(mean((test_data$TARGET_deathRate - test_predictions_v2)^2))

# Prepare data for glmnet
x_train <- model.matrix(TARGET_deathRate ~ . - 1, data = train_data)
y_train <- train_data$TARGET_deathRate
x_test <- model.matrix(TARGET_deathRate ~ . - 1, data = test_data)
y_test <- test_data$TARGET_deathRate

# Ridge Regression
ridge_model_v2 <- cv.glmnet(x_train, y_train, alpha = 0)
ridge_best_lambda_v2 <- ridge_model_v2$lambda.min
ridge_predictions_v2 <- predict(ridge_model_v2, s = ridge_best_lambda_v2, newx = x_test)
ridge_rsq_v2 <- rsq(y_test, ridge_predictions_v2)
ridge_rmse_v2 <- sqrt(mean((y_test - ridge_predictions_v2)^2))

# Lasso Regression
lasso_model_v2 <- cv.glmnet(x_train, y_train, alpha = 1)
lasso_best_lambda_v2 <- lasso_model_v2$lambda.min
lasso_predictions_v2 <- predict(lasso_model_v2, s = lasso_best_lambda_v2, newx = x_test)
lasso_rsq_v2 <- rsq(y_test, lasso_predictions_v2)
lasso_rmse_v2 <- sqrt(mean((y_test - lasso_predictions_v2)^2))

# Compare the results
results_v2 <- list(
  Enhanced_Model_v2 = list(R_squared = test_rsq_v2, RMSE = test_rmse_v2),
  Ridge_v2 = list(R_squared = ridge_rsq_v2, RMSE = ridge_rmse_v2),
  Lasso_v2 = list(R_squared = lasso_rsq_v2, RMSE = lasso_rmse_v2)
)
results_v2