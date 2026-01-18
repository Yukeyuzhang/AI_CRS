### Unsupervised clustering
```r
library("Biobase")
library("BiocGenerics")
library("ConsensusClusterPlus")
library("ggplot2")
library("reticulate")
library("Matrix")
library("readxl")

### Data Import -------------------------------------------------------------------------------------------------
data <- read_excel("E:/Radiomic feature.xlsx", col_names = TRUE)

# Convert tibble to regular data frame
sample_feature_matrix <- as.data.frame(data)

# Set first column as row names
rownames(sample_feature_matrix) <- sample_feature_matrix[[1]]
sample_feature_matrix <- sample_feature_matrix[-1]

# Select all columns except the first and convert them to numeric
sample_feature_matrix[, -1] <- lapply(sample_feature_matrix[, -1], as.numeric)
sample_feature_matrix <- as.matrix(sample_feature_matrix)
sample_feature_matrix <- t(sample_feature_matrix)

### Set parameters for ConsensusClusterPlus function ------------------------------------------------------------------------
# pItem_values <- c(0.9)
# seeds <- 1:40
# title_base <- 'E:/ConsensusCluster'
# 
# for (pItem in pItem_values) {
#   for (seed in seeds) {
#     # Set output path
#     title <- sprintf("%s/pItem_%0.2f_seed_%02d", title_base, pItem, seed)
# 
#     # Create directory (if it doesn't exist)
#     if (!dir.exists(title)) {
#       dir.create(title, recursive = TRUE)
#     }
# 
#     # Run ConsensusClusterPlus
#     results <- ConsensusClusterPlus(sample_feature_matrix,
#                                     maxK = 10,
#                                     reps = 10,
#                                     pItem = pItem,
#                                     pFeature = 1,
#                                     title = title,
#                                     clusterAlg = 'pam',
#                                     distance = 'pearson',
#                                     seed = seed,
#                                     plot = 'png')
# 
#     # Optional: Save results object (for further analysis)
#     save(results, file = sprintf("%s/results.RData", title))
#   }
# }

title <- 'E:/ConsensusCluster'
results <- ConsensusClusterPlus(sample_feature_matrix,
                                maxK = 10,
                                reps = 10,
                                pItem = 0.9,
                                pFeature = 1,
                                title = title,
                                clusterAlg = 'pam',
                                distance = 'pearson',
                                seed = 21,
                                plot = 'pdf')
consensusClass <- results[[4]][["consensusClass"]]
write.csv(consensusClass, file = "E:/ConsensusCluster/cluster group.csv", row.names = FALSE)

# Clustering heatmap adjustment
# dev.off()
# pdf("E:/ConsensusCluster/consensus.pdf")
# consensusMatrix = results[[4]][["consensusMatrix"]]
# heatmap(consensusMatrix,
#         col = colorRampPalette(c("blue", "white","red"))(100),
#         main = "Consensus Matrix",
#         xlab = "Sample Index",
#         ylab = "Sample Index")
# dev.off()

### Cluster Consensus (CLC) and Item Consensus (IC) -----------------------------------------
icl <- calcICL(results,
               title = 'ICL',
               plot = "png")
# dim(icl[["clusterConsensus"]]) # Returns a list with two elements
# icl[["clusterConsensus"]]
# dim(icl[["itemConsensus"]]) # Returns a list with two elements
# icl[["itemConsensus"]][1:5,]
```

## 3D PCA
```r
# Load necessary libraries
library(readxl)
library(dplyr)
library(FactoMineR)  
# library(Biobase)
# library(BiocGenerics)
library(ConsensusClusterPlus)
library(ggplot2)
library(reticulate)
library(Matrix)
library(plotly)

# Read data
train_features <- read_excel("E:/Train Radiomics Features.xlsx")
val_features <- read_excel("E:/Test Radiomics Features.xlsx")
cluster_data <- read_excel("E:/Train group.xlsx")

# Process training data
merged_data <- train_features %>% inner_join(cluster_data, by = "Name")
pca_data <- merged_data %>% select(-Name,-Cluster)

# Perform PCA analysis
pca_result <- PCA(pca_data, graph = FALSE)

# Extract the first 5 principal components
pca_scores <- as.data.frame(pca_result$ind$coord)
pca_scores <- pca_scores[, 1:5]

# Flip the Z-axis
pca_scores$Dim.2 <- -pca_scores$Dim.2
# pca_scores$Dim.3 <- -pca_scores$Dim.3

# Define cluster colors
cluster_colors <- c("Cluster1" = "#A6CEE3",
                    "Cluster2" = "#B2DF8A",
                    "Cluster3" = "#1F78B4",
                    "Cluster4" = "#33A02C")

# Calculate the mean of all principal components for each phenotype
pca_scores$Cluster <- merged_data$Cluster
mean_vectors <- pca_scores %>% 
  group_by(Cluster) %>% 
  summarise(across(everything(), mean, na.rm = TRUE))

# Create 3D PCA plot and add centroids for each cluster
pca_scores$Name <- merged_data$Name
p <- plot_ly(pca_scores, x = ~Dim.1, y = ~Dim.2, z = ~Dim.3, 
             color = ~Cluster, colors = cluster_colors, 
             text = ~Name, 
             type = 'scatter3d', 
             mode = 'markers',
             marker = list(size = 6)) %>%  # Increase marker size
  layout(scene = list(xaxis = list(title = 'PC1'),
                      yaxis = list(title = 'PC2'),
                      zaxis = list(title = 'PC3'),
                      camera = list(eye = list(x = 0.3, y = 1, z = 2))),
         legend = list(itemsizing = 'constant', font = list(size = 20)))  # Control legend size
  # add_trace(data = mean_vectors, x = ~Dim.1, y = ~Dim.2, z = ~Dim.3,
  #           type = 'scatter3d',
  #           mode = 'markers+text',
  #           marker = list(size = 6, color = 'black'),  # Set centroid marker size and color
  #           text = "",  # Display phenotype name
  #           textposition = 'top center')  # Text position

# Display plot
p

# Extract group information (ensure column names are correct)
train_features <- read_excel("E:/Train Radiomics Features.xlsx")
val_features <- read_excel("E:/Test Radiomics Features.xlsx")
cluster_data <- read_excel("E:/Train group.xlsx")

train_features$Cluster <- cluster_data$Cluster  # Adjust based on actual column name

# Calculate centroids for each cluster
centroids <- train_features %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE), .groups = 'drop')  # Calculate centroids for numeric columns only

# Convert centroids to matrix
centroid_matrix <- as.matrix(centroids[, -1])  # Remove cluster column

# Calculate Euclidean distance from each validation sample to each cluster centroid
distance_matrix <- sapply(1:nrow(val_features), function(i) {
  sqrt(colSums((t(centroid_matrix) - as.numeric(val_features[i, -1]))^2))
})

# Determine nearest cluster
predicted_groups <- apply(distance_matrix, 2, which.min)

# Save results
write.csv(predicted_groups, "E:/Test group.csv", row.names = FALSE)
```

```r
library(readxl)
library(dplyr)
library(openxlsx)
library(FactoMineR)  
library(plotly)

# Read feature data
test_features <- read_excel("E:/Test Radiomics Feature.xlsx")
cluster_data <- read_excel("E:/Test group.xlsx")

# Process test data
pca_data <- test_features %>% select(-Name)

# Perform PCA analysis
pca_result <- PCA(pca_data, graph = FALSE)

# Extract the first 3 principal components
pca_scores <- as.data.frame(pca_result$ind$coord)
pca_scores <- pca_scores[, 1:3]

# Flip Z-axis
# pca_scores$Dim.1 <- -pca_scores$Dim.1
pca_scores$Dim.2 <- -pca_scores$Dim.2
# pca_scores$Dim.3 <- -pca_scores$Dim.3

# Define cluster colors
cluster_colors <- c("Cluster1" = "#A6CEE3",
                    "Cluster2" = "#B2DF8A",
                    "Cluster3" = "#1F78B4",
                    "Cluster4" = "#33A02C")

# Calculate the median of all principal components for each phenotype
pca_scores$Cluster <- cluster_data$Cluster
mean_vectors <- pca_scores %>% 
  group_by(Cluster) %>% 
  summarise(across(everything(), median, na.rm = TRUE))

# Create 3D PCA plot and add median points for each cluster
pca_scores$Name <- cluster_data$Name
p <- plot_ly(pca_scores, x = ~Dim.1, y = ~Dim.2, z = ~Dim.3, 
             color = ~Cluster, colors = cluster_colors, 
             text = ~Name, 
             type = 'scatter3d', 
             mode = 'markers',
             marker = list(size = 4)) %>%  # Set marker size
  layout(scene = list(xaxis = list(title = 'PC1'),
                      yaxis = list(title = 'PC2'),
                      zaxis = list(title = 'PC3'),
                      camera = list(eye = list(x = 1, y = 1, z = 0)))) #%>%
  # add_trace(data = mean_vectors, x = ~Dim.1, y = ~Dim.2, z = ~Dim.3,
  #           type = 'scatter3d',
  #           mode = 'markers+text',
  #           marker = list(size = 6, color = 'black'),  # Set median point size and color
  #           text = ~Cluster,  # Display cluster name
  #           textposition = 'top center')  # Text position

# Display the plot
p
```

## <font style="color:rgba(0, 0, 0, 0.8);">Differential protein analysis</font>
```r
# Load necessary R packages
library(readxl)
library(limma)
library(dplyr)
library(ComplexHeatmap)
library(readr)
library(writexl)
library(ggplot2)
library(cowplot)
library(ggrepel) 
library(tibble) 

# Read data
expression_data <- read_excel("E:/Protein.xlsx")

# Convert data to a format suitable for limma
expression_data <- as.data.frame(expression_data)
rownames(expression_data) <- expression_data[[2]]
group_data <- expression_data[["Cluster"]]
expression_data <- t(expression_data[,-c(1, 2, 3)])

# Define comparison combinations
combos <- list(
  list(c("Cluster1"), c("Cluster2")), 
  list(c("Cluster1"), c("Cluster3")),
  list(c("Cluster1"), c("Cluster4")), 
  list(c("Cluster2"), c("Cluster3")),
  list(c("Cluster2"), c("Cluster4")), 
  list(c("Cluster3"), c("Cluster4"))
)

# Loop through each combination
for (combo in combos) {
  group1_clusters <- combo[[1]]
  group2_clusters <- combo[[2]]
  
  # Merge specified clusters into groups
  new_group <- ifelse(group_data %in% group1_clusters, "Group1",
                      ifelse(group_data %in% group2_clusters, "Group2", NA))
  
  # Remove NA values
  valid_indices <- !is.na(new_group)  # Get indices of non-NA values
  filtered_data <- expression_data[ ,valid_indices]  # Filter data using valid_indices
  filtered_group <- new_group[valid_indices]  # Filter group labels
  
  # Perform differential analysis for each combination
  design <- model.matrix(~factor(filtered_group) + 0)
  colnames(design) <- c("Group1", "Group2")
  
  # Calculate the model
  fit <- lmFit(filtered_data, design)
  contrast_name <- "Group1 - Group2"
  contrast_matrix <- makeContrasts(contrasts = contrast_name, levels = design)
  fit_contrast <- contrasts.fit(fit, contrast_matrix)
  fit_eBayes <- eBayes(fit_contrast)
  temp_output <- topTable(fit_eBayes, n = Inf, adjust = "fdr")
  
  # Save differential analysis results
  group1_name <- paste(group1_clusters, collapse = "_")
  group2_name <- paste(group2_clusters, collapse = "_")
  output_file <- paste0("E:/Cluster/1 VS 1/Limma_", group1_name, "_vs_", group2_name, ".csv")
  temp_output <- temp_output %>%
    rownames_to_column(var = "Protein")  # Convert row names to "Protein" column
  write.csv(temp_output, file = output_file, row.names = FALSE)
  
  # Filter and save significant results (P-value < 0.05 and absolute logFC ≥ 1)
  sig_output <- temp_output %>% filter(P.Value < 0.05 & abs(logFC/log(2)) > 1)
  sig_output_file <- paste0("E:/Cluster/1 VS 1/LimmaSig_", group1_name, "_vs_", group2_name, ".csv")
  write.csv(sig_output, file = sig_output_file, row.names = FALSE)
}

# Volcano plot generation function
plot_volcano <- function(data, title) {
  ggplot(data, aes(x = logFC/log(2), y = -log10(P.Value))) +
    geom_point(aes(color = ifelse(P.Value < 0.05 & logFC/log(2) > 1, "Red",
                                  ifelse(P.Value < 0.05 & logFC/log(2) < -1, "Blue", "Gray")),
                   shape = "circle"), alpha = 1) +  # Opaque points
    scale_color_manual(values = c("Red" = "red", "Blue" = "blue", "Gray" = "gray")) +
    labs(title = title, x = "logFC", y = "-log10(P.Value)") +
    xlim(c(-5, 5)) +  # Set x-axis range to ensure data fits within bounds
    theme_minimal() +
    theme(legend.position = "none",  # Hide legend
          panel.background = element_rect(fill = "white"),  # White background
          plot.background = element_rect(fill = "white"),   # White background
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),  # Remove panel border
          plot.margin = margin(5, 10, 5, 10)) +  # Add margins
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +  # P.Value = 0.05 threshold
    geom_vline(xintercept = 1, linetype = "dashed", color = "black") +  # logFC = 1 threshold
    geom_vline(xintercept = -1, linetype = "dashed", color = "black") +  # logFC = -1 threshold
    geom_text_repel(data = subset(data, P.Value < 0.05 & abs(logFC/log(2)) >= 1),
                    aes(label = Protein),  # Use protein names as labels
                    size = 1,  # Label font size
                    box.padding = 0.1,  # Padding around labels
                    point.padding = 0.1,  # Padding between points and labels
                    segment.color = 'black',  # Color of label connecting lines
                    max.overlaps = Inf,  # Allow unlimited label overlaps
                    nudge_x = 0.1,  # Fine-tune label position
                    nudge_y = 0.1)  # Fine-tune label position
}

# Read all results and generate volcano plots
plot_list <- list()

for (combo in combos) {
  group1_clusters <- combo[[1]]
  group2_clusters <- combo[[2]]
  
  # Read differential analysis results
  group1_name <- paste(group1_clusters, collapse = "_")
  group2_name <- paste(group2_clusters, collapse = "_")
  result_file <- paste0("E:/Cluster/1 VS 1/Limma_", group1_name, "_vs_", group2_name, ".csv")
  result_data <- read.csv(result_file)
  
  # Generate volcano plot
  plot_title <- paste0(group1_name, " vs ", group2_name)
  volcano_plot <- plot_volcano(result_data, plot_title)
  plot_list[[plot_title]] <- volcano_plot
}

# Combine all volcano plots into a single file
combined_plot <- plot_grid(plotlist = plot_list, ncol = 3)  # Display 3 plots per row

# Save the combined image
ggsave(filename = "E:/Cluster/1 VS 1/Volcano.png",
       plot = combined_plot,
       width = 14, height = 8, dpi = 300, bg = "white")  # Ensure the image is not cropped
```

