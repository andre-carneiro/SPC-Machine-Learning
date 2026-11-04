# Import libraries
library(rpart)
library(rpart.plot)
library(dplyr)

# Read dataframe and replace dots with spaces in column names
data <- read.csv("data/features_regression_rpart.csv")
colnames(data) <- gsub("\\.", " ", colnames(data))  # Replace dots with spaces

# Function of rpart with an option to use the whole dataset
analyze_tree <- function(data, period_value = NULL, use_whole_dataset = FALSE) {
  
  # Determine the data to use (filtered or entire dataset)
  if (use_whole_dataset) {
    tree_data <- data
    plot_title <- "Decision Tree for whole dataset"
  } else {
    tree_data <- data %>% filter(Period == period_value)
    plot_title <- paste("Decision Tree for", period_value, "period")
  }
  
  # Change data type for 'Time Interval' in the selected data
  #tree_data$`Time Interval` <- as.character(tree_data$`Time Interval`)
  
  # Fit the decision tree model on the selected data
  tree_model <- rpart(`Score Test` ~ `H4 Manual feature selection` + `H5 Feature engineering` + `H2 Correlation removal` + 
                        `H3 Dimensionality reduction` + `H7 Rebalancing Strategy` + `H12 ML model`, 
                      data = tree_data, method = "anova",
                      control = rpart.control(maxdepth = 4))
  
  #plot cp
  plotcp(tree_model)
  
  # Plot the decision tree with a dynamic title
  rpart.plot(tree_model, main = plot_title, 
             digit = 2, type = 4)
  
  # Print summary of the model
  print(summary(tree_model))
  
  # Display other meaningful information
  printcp(tree_model) # Complexity parameter table
  cat("\nVariable Importance:\n")
  print(tree_model$variable.importance) # Variable importance
  
  return(tree_model)
}

# Run the function for different periods
tree_first <- analyze_tree(data, period_value = "first")

tree_second <- analyze_tree(data, period_value = "second")

tree_third <- analyze_tree(data, period_value = "third")

# Run the function for the whole dataset
tree_whole <- analyze_tree(data, use_whole_dataset = TRUE)
