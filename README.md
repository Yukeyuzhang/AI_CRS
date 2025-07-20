Chronic Rhinosinusitis (CRS) Surgical Outcome Predict
This repository contains implementations of noninvasive radiomic and deep learning (DL) models for predicting surgical outcomes in patients with Chronic Rhinosinusitis (CRS) using preoperative sinus CT images.
CRS has a high postoperative recurrence rate, making accurate preoperative prognosis crucial for effective management. This work develops two types of models:
i) A radiomic model using features extracted from semi-automatic (threshold-based) segmentation of CT images
ii) A deep learning model using features from fully-automatic (nnUNetV2-based) segmentation of CT images
Both models demonstrate superior performance compared to traditional prognostic methods, including CT-score models and clinical blood eosinophil-based models. They show stable performance across internal and external validation cohorts.

Key Features
i) Noninvasive Prognosis: Utilizes preoperative CT images, avoiding the need for invasive procedures
ii) Multi-cohort evaluation: Developed and evaluated across multiple patient cohorts from various institutions
iii) Biological Relevance: Multi-omics analysis links radiomic features to tissue proteomic and histologic profiles
iv) Clinical Utility: Post-hoc analysis indicates the models can help stratify patients for optimal surgical approach (extended endoscopic sinus surgery vs. functional endoscopic sinus surgery)

Applications
These models provide preoperative, noninvasive guidance for CRS management by predicting surgical outcomes and recurrence risk, aiding clinicians in making informed treatment decisions.

Model Details
i) Radiomic model: Built using XGBoost with radiomic feature extraction
ii) Deep learning model: Based on ResNet architecture with DL feature extraction
For implementation details, please refer to the source code and documentation in the repository.
