# Vector versions of the regression trees of the configuration experiment (Figure S1, R2-m7).
# Same model as code/original_pipeline/reg_tree_correct.R (rpart, anova, maxdepth = 4); one full-width PDF per period.
# Run from v2/: Rscript code/revision/fig_regression_trees.R
library(rpart)
library(rpart.plot)

data <- read.csv("data/raw/features_regression_rpart.csv", check.names = FALSE)[, -1]
for (period in c("first", "second", "third")) {
  d <- data[data$Period == period, ]
  tree <- rpart(`Score Test` ~ `H4 Manual feature selection` + `H5 Feature engineering` + `H2 Correlation removal` +
                  `H3 Dimensionality reduction` + `H7 Rebalancing Strategy` + `H12 ML model`,
                data = d, method = "anova", control = rpart.control(maxdepth = 4))
  imp <- tree$variable.importance
  cat(period, ":", paste(names(imp), round(imp / sum(imp), 3), sep = " = ", collapse = "; "), "\n")
  for (out in c(file.path("figures", paste0("regression_tree_", period, ".pdf")),
                file.path("manuscript", "Paper", "Images", paste0("regression_tree_", period, ".pdf")))) {
    pdf(out, width = 14, height = 6.5, family = "Times")
    rpart.plot(tree, digits = 3, type = 2, extra = 101, under = TRUE, fallen.leaves = FALSE,
               box.palette = "Grays", shadow.col = 0, branch.lty = 1, varlen = 0, faclen = 0,
               tweak = 1.1, space = 0.4, gap = 0.6)
    dev.off()
  }
}
