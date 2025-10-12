# nnU-Net Prostate Segmentation Pipeline

This repository contains an end-to-end **medical image segmentation workflow** for prostate MRI, based on **nnU-Net v2**.  
The project includes dataset setup, preprocessing, model training, cross-validation, inference, and quality control visualization.

---

## 📁 Project Structure
medicalimaging/nnunet/
│
├── nnUNet/ # nnU-Net v2 source and scripts
├── results/ # evaluation and quality control summaries
├── dockerfile  # container definition for reproducibility
└── .gitignore


# Project Summary

This project develops a complete medical image segmentation pipeline for prostate MRI using nnU-Net v2 and extends it with synthetic data generation via diffusion models to improve generalization.

We began by organizing the Dataset001_PROSTATE based on the LUND probe dataset - https://datahub.aida.scilifelab.se/10.23698/aida/lund-probe -  in nnU-Net format (imagesTr, labelsTr, imagesTs) and verifying dataset integrity. Using Docker, nnU-Net was configured with environment variables for raw, preprocessed, and trained model directories. The pipeline was trained using 5-fold cross-validation (nnUNetv2_train 1 3d_fullres 0–4) and validated automatically. Training logs, checkpoints (checkpoint_best.pth), and model plans were saved under /data/nnUNet_trained_models/.

After training, we performed inference on the test set using an ensemble of all folds (nnUNetv2_predict … -f all) to generate 3D segmentation masks for 130 test cases. Quantitative evaluation was performed with nnUNetv2_evaluate_folder, producing Dice ≈ 0.90 and IoU ≈ 0.82. Cross-validation summaries were extracted (summary.json) and compared to test results to confirm consistency.

For qualitative quality control (QC), a Python visualization tool using SimpleITK and matplotlib was developed to overlay predicted and ground truth segmentations on MRI slices. The outputs (copied to quality_control_pngs) were used for visual verification of segmentation quality.
The next phase will integrate a diffusion model for synthetic MRI generation, followed by fine-tuning nnU-Net on mixed real + synthetic data, and comprehensive evaluation on real-world test cases.

This work establishes a reproducible foundation for automated prostate MRI segmentation and data augmentation research using nnU-Net and diffusion models.
