<h2 id="KAmaP"><font style="color:rgba(0, 0, 0, 0.95);">Environment configuration</font></h2>
```python
### Activate the base environment
conda init
source ~/.bashrc
conda activate nnunet_env

### Configure the system
conda create -n nnunet_env python=3.10
conda activate nnunet_env
conda deactivate

### Install packages
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install nnunet
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
```

<h2 id="eoSpl"><font style="color:rgba(0, 0, 0, 0.95);">nnUNet segementation</font></h2>
```python
nnUNetv2_convert_MSD_dataset -i /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Task333_CRS -overwrite_id 333

nnUNetv2_plan_and_preprocess -d 333 --verify_dataset_integrity

#rm -r /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_preprocessed/Dataset333_CRS/nnUNetTrainer_nnUNetPlans_2d/fold_0

nnUNetv2_train 333 2d 0 --val --npz
nnUNetv2_train 333 2d 1 --val --npz 
nnUNetv2_train 333 2d 2 --val --npz
nnUNetv2_train 333 2d 3 --val --npz
nnUNetv2_train 333 2d 4 --val --npz

nnUNetv2_train 333 3d_lowres 0 --val --npz
nnUNetv2_train 333 3d_lowres 1 --val --npz
nnUNetv2_train 333 3d_lowres 2 --val --npz
nnUNetv2_train 333 3d_lowres 3 --val --npz
nnUNetv2_train 333 3d_lowres 4 --val --npz

nnUNetv2_train 333 3d_fullres 0 --val --npz
nnUNetv2_train 333 3d_fullres 1 --val --npz
nnUNetv2_train 333 3d_fullres 2 --val --npz
nnUNetv2_train 333 3d_fullres 3 --val --npz
nnUNetv2_train 333 3d_fullres 4 --val --npz

nnUNetv2_train 333 3d_cascade_fullres 0 --val --npz
nnUNetv2_train 333 3d_cascade_fullres 1 --val --npz
nnUNetv2_train 333 3d_cascade_fullres 2 --val --npz
nnUNetv2_train 333 3d_cascade_fullres 3 --val --npz
nnUNetv2_train 333 3d_cascade_fullres 4 --val --npz

nnUNetv2_find_best_configuration 333 -c 2d 3d_fullres 3d_lowres 3d_cascade_fullres -f 0 1 2 3 4
nnUNetv2_predict -d Dataset333_CRS -i /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset333_CRS/imagesTs -o /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset333_2d_predict -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans --save_probabilities
nnUNetv2_predict -d Dataset333_CRS -i /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset333_CRS/imagesTs -o /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset333_3d_fullres_predict -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans --save_probabilities

nnUNetv2_find_best_configuration 333 -c 2d 3d_fullres 3d_lowres 3d_cascade_fullres -f 0 1 2 3 4

#***All results:***
#nnUNetTrainer__nnUNetPlans__2d: 0.9476568510953605
#nnUNetTrainer__nnUNetPlans__3d_fullres: 0.9456064427159295
#nnUNetTrainer__nnUNetPlans__3d_lowres: 0.9442375045435938
#nnUNetTrainer__nnUNetPlans__3d_cascade_fullres: 0.945518072172821
#ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_fullres___0_1_2_3_4: 0.9484509399143581
#ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_lowres___0_1_2_3_4: 0.9482162819241774
#ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_cascade_fullres___0_1_2_3_4: 0.9483575119013308
#ensemble___nnUNetTrainer__nnUNetPlans__3d_fullres___nnUNetTrainer__nnUNetPlans__3d_lowres___0_1_2_3_4: 0.945577037619692
#ensemble___nnUNetTrainer__nnUNetPlans__3d_fullres___nnUNetTrainer__nnUNetPlans__3d_cascade_fullres___0_1_2_3_4: 0.9459293397308728
#ensemble___nnUNetTrainer__nnUNetPlans__3d_lowres___nnUNetTrainer__nnUNetPlans__3d_cascade_fullres___0_1_2_3_4: 0.9451434404596001

#*Best*: ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_fullres___0_1_2_3_4: 0.9484509399143581

#***Determining postprocessing for best model/ensemble***
#Removing all but the largest foreground region did not improve results!

#***Run inference like this:***

#An ensemble won! What a surprise! Run the following commands to run predictions with the ensemble members:

#nnUNetv2_predict -d Dataset333_CRS -i INPUT_FOLDER -o OUTPUT_FOLDER_MODEL_1 -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans --save_probabilities
#nnUNetv2_predict -d Dataset333_CRS -i INPUT_FOLDER -o OUTPUT_FOLDER_MODEL_2 -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans --save_probabilities

#The run ensembling with:

#nnUNetv2_ensemble -i OUTPUT_FOLDER_MODEL_1 OUTPUT_FOLDER_MODEL_2 -o OUTPUT_FOLDER -np 8

***Once inference is completed, run postprocessing like this:***
nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset333_CRS/ensembles/ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_fullres___0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset333_CRS/ensembles/ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_fullres___0_1_2_3_4/plans.json

nnUNetv2_predict -d Dataset333_CRS -i /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset333_CRS/imagesTs -o /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/Dataset333_2d_predict -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans --save_probabilities
/new_hme/zkz/nnUNet/nnUNetFrame/DATASET/Dataset333_2d_predict
nnUNetv2_predict -d Dataset333_CRS -i /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset333_CRS/imagesTs -o /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/Dataset333_3d_fullres_predict -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans --save_probabilities
nnUNetv2_ensemble -i /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/Dataset333_2d_predict /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/Dataset333_3d_fullres_predict -o /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/Dataset333_Ensemble -np 8
nnUNetv2_apply_postprocessing -i /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/Dataset333_Ensemble -o /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/Dataset333_Ensemble_PP -pp_pkl_file /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset333_CRS/ensembles/ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_fullres___0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /new_hme/zkz/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset333_CRS/ensembles/ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_fullres___0_1_2_3_4/plans.json
```

<h2 id="JEDBi">Train-val data evaluation</h2>
```python
import os
import json
import pandas as pd

# Define the root directory and the path for the output Excel file
root_dir = r'H:\Data1'
output_excel_path = r'H:\Train-val.xlsx'

# Initialize an empty DataFrame to store the results
df = pd.DataFrame(columns=['Filename', 'Dice', 'IoU'])

# Traverse all subfolders in the root directory
for subdir, _, _ in os.walk(root_dir):
    validation_dir = os.path.join(subdir, 'validation')
    summary_file = os.path.join(validation_dir, 'summary.json')

    # Check if the summary.json file exists
    if os.path.exists(summary_file):
        # Print the file path for verification
        print(f"Processing file: {summary_file}")

        # Read the summary.json file
        with open(summary_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file: {summary_file}, error: {e}")
                continue

        # Print the contents of the summary.json file to check its structure
        print(f"Contents of {summary_file}: {json.dumps(data, indent=2)}")

        # Extract data from the metric_per_case array
        for case in data.get('metric_per_case', []):
            prediction_file = case.get('prediction_file')
            dice = case.get('metrics', {}).get('1', {}).get('Dice')
            iou = case.get('metrics', {}).get('1', {}).get('IoU')

            # Print the extracted data for verification
            print(f"Extracted data - prediction_file: {prediction_file}, Dice: {dice}, IoU: {iou}")

            if prediction_file is None or dice is None or iou is None:
                print(f"Missing data in file: {summary_file}")
                continue

            # Extract the filename (the last part of the path)
            filename = os.path.basename(prediction_file).replace('.nii.gz', '')

            # Add the data to the DataFrame
            df = df.append({'Filename': filename, 'Dice': dice, 'IoU': iou}, ignore_index=True)

# Save the DataFrame to an Excel file
df.to_excel(output_excel_path, index=False)
print(f"Data saved to {output_excel_path}")
```

<h2 id="LMMFD">Test data evaluation</h2>
```python
import os
import nibabel as nib
import numpy as np
import pandas as pd

def load_nifti_image(file_path):
    """Load a NIfTI image from the specified file path."""
    try:
        img = nib.load(file_path)
        img_data = img.get_fdata()
        return img_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def dice_and_iou(y_true, y_pred):
    """Calculate Dice coefficient and Intersection over Union (IoU) between two binary masks."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    
    dice = 2. * intersection.sum() / (y_true.sum() + y_pred.sum())
    iou = intersection.sum() / union.sum()

    return dice, iou

def calculate_dice_and_iou_for_all_pairs(true_dir, pred_dir, output_csv, max_pairs=100):
    """Calculate Dice and IoU for all pairs of true and predicted NIfTI images in specified directories."""
    true_files = sorted([f for f in os.listdir(true_dir) if f.endswith('.nii.gz')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])
    
    results = []
    count = 0
    
    for true_file in true_files:
        if count >= max_pairs:
            break

        pred_file = true_file  # Assume filenames are identical for pairing
        true_path = os.path.join(true_dir, true_file)
        pred_path = os.path.join(pred_dir, pred_file)
        print(true_file)
        
        if os.path.exists(pred_path):
            y_true = load_nifti_image(true_path)
            y_pred = load_nifti_image(pred_path)
            
            if y_true is not None and y_pred is not None:
                dice, iou = dice_and_iou(y_true, y_pred)
                results.append({'File': true_file, 'Dice': dice, 'IoU': iou})
            else:
                results.append({'File': true_file, 'Dice': 'Error', 'IoU': 'Error'})
        else:
            print(f"Warning: {pred_file} not found in prediction directory")
            results.append({'File': true_file, 'Dice': 'File not found', 'IoU': 'File not found'})
        
        count += 1
    
    # Save results to CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Define directory paths and output CSV file path
true_dir = r'H:\Data2'
pred_dir = r'H:\Data3'
output_csv = r'H:\Test.csv'

# Calculate and save Dice coefficient and IoU (process first 100 pairs only)
calculate_dice_and_iou_for_all_pairs(true_dir, pred_dir, output_csv)#, max_pairs=10)
```

<h2 id="JqgU1"><font style="color:rgb(28, 31, 35);">Dice_iou_visualization</font></h2>
```r
# Load necessary libraries
library(read.csv)   # For reading CSV files
library(ggplot2)   # For data visualization
library(dplyr)     # For data manipulation
library(scales)    # For axis formatting
library(gridExtra) # For combining plots
library(RColorBrewer) # For color palettes

# Set up Chinese font support
windowsFonts(myFont = windowsFont("SimHei"))

# Define configuration parameters
config <- list(
  plot_width = 10,
  plot_height = 12,
  random_seed = 123,
  dice_range = c(0.80, 1.00),
  iou_range = c(0.65, 1.00),
  point_size = 2.5,
  title_size = 16,
  axis_text_size = 12,
  axis_title_size = 14,
  grid_alpha = 0.3,
  n_bins = 30,
  color_scheme = "Set1"
)

# Function to load CSV data
load_data <- function(file_path) {
  tryCatch({
    message("Loading data from: ", file_path)
    data <- read.csv(file_path, stringsAsFactors = FALSE)
    
    if (nrow(data) == 0) {
      warning("Loaded data is empty")
    }
    
    message("Successfully loaded ", nrow(data), " records")
    return(data)
  }, error = function(e) {
    stop(paste("Failed to load data:", e$message))
  })
}

# Function to preprocess data
preprocess_data <- function(data, seed) {
  message("Preprocessing data...")
  
  # Ensure consistent column names
  colnames(data) <- c("Sample", "Dice", "IoU")
  
  # Data cleaning and type conversion
  data <- data %>%
    mutate(
      Dice = as.numeric(Dice),
      IoU = as.numeric(IoU)
    ) %>%
    filter(!is.na(Dice), !is.na(IoU))
  
  # Randomly shuffle sample order
  set.seed(seed)
  data <- data[sample(nrow(data)), ]
  data$Sample <- factor(data$Sample, levels = data$Sample)
  
  message("Preprocessing complete. Valid samples: ", nrow(data))
  return(data)
}

# Function to create scatter plots
create_scatter_plot <- function(data, y_var, title, y_limits, config) {
  # Calculate statistics
  mean_val <- mean(data[[y_var]])
  median_val <- median(data[[y_var]])
  sd_val <- sd(data[[y_var]])
  
  # Create base plot
  p <- ggplot(data, aes_string(x = "Sample", y = y_var)) +
    geom_point(aes(color = ..y..), size = config$point_size, alpha = 0.8) +
    scale_color_gradientn(
      colors = brewer.pal(9, config$color_scheme),
      limits = y_limits,
      guide = guide_colorbar(title = y_var)
    ) +
    geom_hline(yintercept = mean_val, linetype = "dashed", color = "red", size = 0.8) +
    geom_hline(yintercept = median_val, linetype = "dotted", color = "blue", size = 0.8) +
    labs(
      title = title,
      x = "Sample",
      y = y_var
    ) +
    scale_x_discrete(expand = c(0.05, 0)) +
    scale_y_continuous(
      limits = y_limits,
      breaks = seq(y_limits[1], y_limits[2], by = 0.05),
      labels = percent_format(accuracy = 1)
    ) +
    theme_minimal() +
    theme(
      text = element_text(family = "myFont"),
      plot.title = element_text(
        size = config$title_size, 
        face = "bold", 
        hjust = 0.5,
        margin = margin(b = 10)
      ),
      axis.title = element_text(size = config$axis_title_size, face = "bold"),
      axis.text.x = element_blank(),
      axis.text.y = element_text(size = config$axis_text_size),
      axis.ticks.x = element_line(),
      axis.line = element_line(color = "black"),
      panel.grid.major = element_line(color = "gray80", linetype = "dashed", alpha = config$grid_alpha),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white"),
      legend.position = "right",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 10),
      plot.margin = margin(10, 20, 10, 10)
    )
  
  # Add statistics annotation
  p <- p + annotate(
    "text", 
    x = Inf, 
    y = -Inf, 
    hjust = 1.1, 
    vjust = -0.5,
    label = paste0(
      "Mean: ", format(mean_val, digits = 3), "\n",
      "Median: ", format(median_val, digits = 3), "\n",
      "SD: ", format(sd_val, digits = 3)
    ),
    size = 4,
    family = "myFont"
  )
  
  return(p)
}

# Function to create distribution plots
create_distribution_plot <- function(data, var_name, title, config) {
  p <- ggplot(data, aes_string(x = var_name)) +
    geom_histogram(
      aes(y = ..density..),
      bins = config$n_bins,
      color = "black",
      fill = brewer.pal(9, config$color_scheme)[3],
      alpha = 0.7
    ) +
    geom_density(color = "red", size = 1) +
    geom_vline(
      aes(xintercept = mean(!!sym(var_name))),
      linetype = "dashed",
      color = "blue",
      size = 1
    ) +
    labs(
      title = title,
      x = var_name,
      y = "Density"
    ) +
    theme_minimal() +
    theme(
      text = element_text(family = "myFont"),
      plot.title = element_text(
        size = config$title_size - 2, 
        face = "bold", 
        hjust = 0.5
      ),
      axis.title = element_text(size = config$axis_title_size - 2, face = "bold"),
      axis.text = element_text(size = config$axis_text_size - 2),
      panel.grid.major = element_line(color = "gray80", linetype = "dashed", alpha = config$grid_alpha),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white"),
      plot.margin = margin(10, 10, 10, 10)
    )
  
  return(p)
}

# Function to create correlation plot
create_correlation_plot <- function(data, config) {
  # Calculate correlation coefficient
  correlation <- cor(data$Dice, data$IoU)
  
  p <- ggplot(data, aes(x = Dice, y = IoU)) +
    geom_point(aes(color = Sample), size = config$point_size - 0.5, alpha = 0.7) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
    scale_color_manual(values = rainbow(nrow(data))) +
    labs(
      title = "Correlation between Dice and IoU",
      x = "Dice Coefficient",
      y = "IoU Value"
    ) +
    annotate(
      "text", 
      x = min(data$Dice), 
      y = max(data$IoU), 
      hjust = 0, 
      vjust = 1,
      label = paste0("Corr: ", format(correlation, digits = 3)),
      size = 4,
      family = "myFont"
    ) +
    theme_minimal() +
    theme(
      text = element_text(family = "myFont"),
      plot.title = element_text(
        size = config$title_size - 1, 
        face = "bold", 
        hjust = 0.5
      ),
      axis.title = element_text(size = config$axis_title_size - 1, face = "bold"),
      axis.text = element_text(size = config$axis_text_size - 1),
      panel.grid.major = element_line(color = "gray80", linetype = "dashed", alpha = config$grid_alpha),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white"),
      legend.position = "none",
      plot.margin = margin(10, 10, 10, 10)
    )
  
  return(p)
}

# Function to create combined plot
create_combined_plot <- function(data, config) {
  # Create individual plots
  scatter_dice <- create_scatter_plot(
    data, 
    "Dice", 
    "Dice Coefficients by Sample", 
    config$dice_range,
    config
  )
  
  scatter_iou <- create_scatter_plot(
    data, 
    "IoU", 
    "IoU Values by Sample", 
    config$iou_range,
    config
  )
  
  dist_dice <- create_distribution_plot(
    data, 
    "Dice", 
    "Distribution of Dice Coefficients",
    config
  )
  
  dist_iou <- create_distribution_plot(
    data, 
    "IoU", 
    "Distribution of IoU Values",
    config
  )
  
  corr_plot <- create_correlation_plot(data, config)
  
  # Combine plots
  combined_plot <- gridExtra::grid.arrange(
    scatter_dice, dist_dice,
    scatter_iou, dist_iou,
    corr_plot,
    ncol = 2, 
    nrow = 3,
    heights = c(1, 1, 1),
    widths = c(1.5, 1),
    layout_matrix = rbind(
      c(1, 2),
      c(3, 4),
      c(5, 5)
    )
  )
  
  return(combined_plot)
}

# Main processing function
process_dataset <- function(input_path, output_path, config) {
  # Start processing
  start_time <- Sys.time()
  message("===== Starting processing for: ", input_path, " =====")
  
  # Load and process data
  data <- load_data(input_path)
  processed_data <- preprocess_data(data, config$random_seed)
  
  # Create and save visualization
  message("Generating visualizations...")
  pdf(output_path, width = config$plot_width, height = config$plot_height)
  
  combined_plot <- create_combined_plot(processed_data, config)
  print(combined_plot)
  
  dev.off()
  message(paste("Visualization saved to:", output_path))
  
  # Finish processing
  end_time <- Sys.time()
  message(paste("Processing completed. Time elapsed:", format(end_time - start_time, digits = 2)))
  message("===== Processing finished =====")
}

# Define dataset paths
train_val_input <- "H:/Train-val.csv"
test_input <- "H:/Test.csv"

# Define output paths
train_val_output <- "H:/Train-val Results.pdf"
test_output <- "H:/Test Results.pdf"

# Process both datasets
process_dataset(train_val_input, train_val_output, config)
process_dataset(test_input, test_output, config)
```

<h2 id="fmUvM"><font style="color:rgb(51, 51, 51);">版本</font></h2>
<h3 id="WWnwt"><font style="color:rgb(51, 51, 51);">1.python</font></h3>
```python
(nnunet_env) $ python
Python 3.10.15 (main, Oct  3 2024, 07:27:34) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

<h3 id="PFojs"><font style="color:rgb(51, 51, 51);">2.PyTorch</font></h3>
```python
(nnunet_env) $ python
Python 3.10.15 (main, Oct  3 2024, 07:27:34) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
    >>> import torch
    >>> print(torch.cuda.is_available())
    True
    >>> print(torch.__version__)
    2.4.1
```

<h3 id="xALXf"><font style="color:rgb(51, 51, 51);">3.CUDA</font></h3>
```python
(nnunet_env) $ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:10:22_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```

<h3 id="kN5Qe"><font style="color:rgb(51, 51, 51);">4.nnUNetv2</font></h3>
```python
Name: nnunetv2
Version: 2.5.1
Summary: nnU-Net is a framework for out-of-the box image segmentation.
                                Home-page: https://github.com/MIC-DKFZ/nnUNet
                                Author: Helmholtz Imaging Applied Computer Vision Lab
                                Author-email: Fabian Isensee <f.isensee@dkfz-heidelberg.de>
                                License: Apache License
                                Version 2.0, January 2004
                                http://www.apache.org/licenses/

                                TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction,
and distribution as defined by Sections 1 through 9 of this document.

    "Licensor" shall mean the copyright owner or entity authorized by
the copyright owner that is granting the License.

    "Legal Entity" shall mean the union of the acting entity and all
other entities that control, are controlled by, or are under common
control with that entity. For the purposes of this definition,
"control" means (i) the power, direct or indirect, to cause the
direction or management of such entity, whether by contract or
    otherwise, or (ii) ownership of fifty percent (50%) or more of the
outstanding shares, or (iii) beneficial ownership of such entity.

"You" (or "Your") shall mean an individual or Legal Entity
exercising permissions granted by this License.

    "Source" form shall mean the preferred form for making modifications,
including but not limited to software source code, documentation
source, and configuration files.

    "Object" form shall mean any form resulting from mechanical
transformation or translation of a Source form, including but
not limited to compiled object code, generated documentation,
and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or
Object form, made available under the License, as indicated by a
copyright notice that is included in or attached to the work
(an example is provided in the Appendix below).

 "Derivative Works" shall mean any work, whether in Source or Object
form, that is based on (or derived from) the Work and for which the
editorial revisions, annotations, elaborations, or other modifications
represent, as a whole, an original work of authorship. For the purposes
              of this License, Derivative Works shall not include works that remain
              separable from, or merely link (or bind by name) to the interfaces of,
              the Work and Derivative Works thereof.
        
              "Contribution" shall mean any work of authorship, including
              the original version of the Work and any modifications or additions
              to that Work or Derivative Works thereof, that is intentionally
              submitted to Licensor for inclusion in the Work by the copyright owner
              or by an individual or Legal Entity authorized to submit on behalf of
              the copyright owner. For the purposes of this definition, "submitted"
              means any form of electronic, verbal, or written communication sent
              to the Licensor or its representatives, including but not limited to
              communication on electronic mailing lists, source code control systems,
              and issue tracking systems that are managed by, or on behalf of, the
              Licensor for the purpose of discussing and improving the Work, but
              excluding communication that is conspicuously marked or otherwise
              designated in writing by the copyright owner as "Not a Contribution."
        
              "Contributor" shall mean Licensor and any individual or Legal Entity
              on behalf of whom a Contribution has been received by Licensor and
              subsequently incorporated within the Work.
        
           2. Grant of Copyright License. Subject to the terms and conditions of
              this License, each Contributor hereby grants to You a perpetual,
              worldwide, non-exclusive, no-charge, royalty-free, irrevocable
              copyright license to reproduce, prepare Derivative Works of,
              publicly display, publicly perform, sublicense, and distribute the
              Work and such Derivative Works in Source or Object form.
        
           3. Grant of Patent License. Subject to the terms and conditions of
              this License, each Contributor hereby grants to You a perpetual,
              worldwide, non-exclusive, no-charge, royalty-free, irrevocable
              (except as stated in this section) patent license to make, have made,
              use, offer to sell, sell, import, and otherwise transfer the Work,
              where such license applies only to those patent claims licensable
              by such Contributor that are necessarily infringed by their
              Contribution(s) alone or by combination of their Contribution(s)
              with the Work to which such Contribution(s) was submitted. If You
              institute patent litigation against any entity (including a
              cross-claim or counterclaim in a lawsuit) alleging that the Work
              or a Contribution incorporated within the Work constitutes direct
              or contributory patent infringement, then any patent licenses
              granted to You under this License for that Work shall terminate
              as of the date such litigation is filed.
        
           4. Redistribution. You may reproduce and distribute copies of the
              Work or Derivative Works thereof in any medium, with or without
              modifications, and in Source or Object form, provided that You
              meet the following conditions:
        
              (a) You must give any other recipients of the Work or
                  Derivative Works a copy of this License; and
        
              (b) You must cause any modified files to carry prominent notices
                  stating that You changed the files; and
        
              (c) You must retain, in the Source form of any Derivative Works
                  that You distribute, all copyright, patent, trademark, and
                  attribution notices from the Source form of the Work,
                  excluding those notices that do not pertain to any part of
                  the Derivative Works; and
        
              (d) If the Work includes a "NOTICE" text file as part of its
                  distribution, then any Derivative Works that You distribute must
                  include a readable copy of the attribution notices contained
                  within such NOTICE file, excluding those notices that do not
                  pertain to any part of the Derivative Works, in at least one
                  of the following places: within a NOTICE text file distributed
                  as part of the Derivative Works; within the Source form or
                  documentation, if provided along with the Derivative Works; or,
                  within a display generated by the Derivative Works, if and
                  wherever such third-party notices normally appear. The contents
                  of the NOTICE file are for informational purposes only and
                  do not modify the License. You may add Your own attribution
                  notices within Derivative Works that You distribute, alongside
                  or as an addendum to the NOTICE text from the Work, provided
                  that such additional attribution notices cannot be construed
                  as modifying the License.
        
              You may add Your own copyright statement to Your modifications and
              may provide additional or different license terms and conditions
              for use, reproduction, or distribution of Your modifications, or
              for any such Derivative Works as a whole, provided Your use,
              reproduction, and distribution of the Work otherwise complies with
              the conditions stated in this License.
        
           5. Submission of Contributions. Unless You explicitly state otherwise,
              any Contribution intentionally submitted for inclusion in the Work
              by You to the Licensor shall be under the terms and conditions of
              this License, without any additional terms or conditions.
              Notwithstanding the above, nothing herein shall supersede or modify
              the terms of any separate license agreement you may have executed
              with Licensor regarding such Contributions.
        
           6. Trademarks. This License does not grant permission to use the trade
              names, trademarks, service marks, or product names of the Licensor,
              except as required for reasonable and customary use in describing the
              origin of the Work and reproducing the content of the NOTICE file.
        
           7. Disclaimer of Warranty. Unless required by applicable law or
              agreed to in writing, Licensor provides the Work (and each
              Contributor provides its Contributions) on an "AS IS" BASIS,
              WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
              implied, including, without limitation, any warranties or conditions
              of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
              PARTICULAR PURPOSE. You are solely responsible for determining the
              appropriateness of using or redistributing the Work and assume any
              risks associated with Your exercise of permissions under this License.
        
           8. Limitation of Liability. In no event and under no legal theory,
              whether in tort (including negligence), contract, or otherwise,
              unless required by applicable law (such as deliberate and grossly
              negligent acts) or agreed to in writing, shall any Contributor be
              liable to You for damages, including any direct, indirect, special,
              incidental, or consequential damages of any character arising as a
              result of this License or out of the use or inability to use the
              Work (including but not limited to damages for loss of goodwill,
              work stoppage, computer failure or malfunction, or any and all
              other commercial damages or losses), even if such Contributor
              has been advised of the possibility of such damages.
        
           9. Accepting Warranty or Additional Liability. While redistributing
              the Work or Derivative Works thereof, You may choose to offer,
              and charge a fee for, acceptance of support, warranty, indemnity,
              or other liability obligations and/or rights consistent with this
              License. However, in accepting such obligations, You may act only
              on Your own behalf and on Your sole responsibility, not on behalf
              of any other Contributor, and only if You agree to indemnify,
              defend, and hold each Contributor harmless for any liability
              incurred by, or claims asserted against, such Contributor by reason
              of your accepting any such warranty or additional liability.
        
           END OF TERMS AND CONDITIONS
        
           APPENDIX: How to apply the Apache License to your work.
        
              To apply the Apache License to your work, attach the following
              boilerplate notice, with the fields enclosed by brackets "[]"
              replaced with your own identifying information. (Don't include
              the brackets!)  The text should be enclosed in the appropriate
              comment syntax for the file format. We also recommend that a
              file or class name and description of purpose be included on the
              same "printed page" as the copyright notice for easier
              identification within third-party archives.
        
           Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
        
           Licensed under the Apache License, Version 2.0 (the "License");
           you may not use this file except in compliance with the License.
           You may obtain a copy of the License at
        
               http://www.apache.org/licenses/LICENSE-2.0
        
           Unless required by applicable law or agreed to in writing, software
           distributed under the License is distributed on an "AS IS" BASIS,
           WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
           See the License for the specific language governing permissions and
           limitations under the License.
Location: /new_hme/zkz/.conda/envs/nnunet_env/lib/python3.10/site-packages
Editable project location: /new_hme/zkz/nnUNet
Requires: acvl-utils, batchgenerators, batchgeneratorsv2, dicom2nifti, dynamic-network-architectures, einops, graphviz, imagecodecs, matplotlib, nibabel, numpy, pandas, requests, scikit-image, scikit-learn, scipy, seaborn, SimpleITK, tifffile, torch, tqdm, yacs
Required-by:
```

  
 

