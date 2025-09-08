Development and Evaluation of Radiomic and Deep Learning Models Based on Preoperative Sinus CT Images for Predicting Surgical Outcomes in Chronic Rhinosinusitis: A Nationwide Multicenter Study
-------------------------------------------------------------------------------------------------------
Required content
· Compiled standalone software and source code:
This repository provides the source code for implementing noninvasive radiomic and deep learning (DL) models that can predict the surgical prognosis of patients with chronic rhinosinusitis (CRS) using preoperative sinus CT images.
· A small real dataset to demo the software and source code:
Visit https://www.tongjicrsdata.com to obtain a small anonymous dataset (including the real clinical data and preoperative sinus CT images from 50 patients) for one-click demonstration of the software. Meanwhile, download requests are supported, and the dataset can be used to test the source code.

1. System requirements
· All software dependencies and operating systems (including version numbers):
Operating systems: Windows 10 (64-bit), Ubuntu 20.04 LTS (64-bit)
· Versions the software has been tested on:
Core software dependencies: Python: 3.10.15, PyTorch: 2.4.1, nnUNetV2: 2.5.1, R: 4.4.2
Other common scientific computing libraries (numpy, pandas, matplotlib, etc.) with versions compatible with the above.
· Any required non-standard hardware:
It is recommended to use a CUDA-supported GPU (NVIDIA) to accelerate the training and inference of deep learning models.

2. Installation guide
· Instructions:
If readers use the cloud platform, no installation is required. After registration and login, they can directly process sample data. If readers use the relevant code in this GitHub repository, please first install Python and R in the operating system, then follow Markdown Files 1–5 to install the corresponding packages using personal data or public data obtained via cloud platform application.
· Typical install time on a "normal" desktop computer:
If readers use the cloud platform, no typical install is needed. If readers use the relevant code of this GitHub, approximately 10-15 minutes are required for the initial installation, depending on internet speed and hardware configuration (mainly due to downloading and installing large packages such as nnUNetV2 and PyRadiomics).

3. Demo
· Instructions to run on data:
We recommend that readers run the model directly on the cloud platform rather than train it from scratch (of course, you can choose to do so, as all source codes are public).
· Expected run time for demo on a "normal" desktop computer:
The demonstration of a single patient on the cloud platform takes about 30 seconds, depending on network configuration and cloud server parallelism. If readers develop models from scratch, the time consumed depends on the amount of training and validation data, epochs, and computing power configuration.

4. Instructions for use
· How to run the software on your data:
After preparing your preoperative sinus CT image data, ensure the images are in a compatible format. After your user account application is approved, you are welcome to upload your original data for one-click processing. Alternatively, you can use your preoperative sinus CT image data to train the model from scratch. All relevant source codes are provided, allowing you to train the model from scratch using your own data.
· Reproduction instructions:
All data in this study are supported for acquisition and one-click reproduction. Please contact the author at kzzhuu@foxmail.com or zhengliuent@hotmail.com.

Additional information: 
  CRS has a high postoperative recurrence rate, making accurate preoperative prognosis crucial for effective management. This work develops two types of models.
  i) A radiomic model using features extracted from semi-automatic (threshold-based) segmentation of CT images.
  ii) A deep learning model using features from fully-automatic (nnUNetV2-based) segmentation of CT images.
  Both models demonstrate superior performance compared to traditional prognostic methods, including CT-score models and clinical blood eosinophil-based models. They exhibit stable performance across both internal and external test cohorts.

Key Points: 
  i) Noninvasive prognosis: Utilizes preoperative CT images, avoiding the need for invasive procedures.
  ii) Multi-cohort evaluation: Developed and evaluated across the largest CRS patient cohorts from various institutions, demonstrating stable and superior performance.
  iii) Biological relevance: Multi-omics analysis links radiomic features to tissue proteomic and histologic profiles.
  iv) Clinical utility: Clinical intervention analysis indicates the models can help stratify patients for optimal surgical approach (extended endoscopic sinus surgery vs. functional endoscopic sinus surgery).
