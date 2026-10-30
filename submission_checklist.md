# HCLTech Career Shaper Submission Checklist
### Project 35: Crop Disease Detection Using Convolutional Neural Networks
**Academic Year:** 2025–26 | **Programme:** HCLTech Career Shaper III Year Industry Project

---

## OFFICIAL DELIVERABLES CHECKLIST (WEEK 4 DEADLINE)

Use this checklist to verify that all mandatory submission requirements are complete before final portal upload:

| Deliverable Item | Required Format / Specification | Project File Location | Status |
| :--- | :--- | :--- | :---: |
| **1. Completed Working Application** | Interactive prototype for leaf image upload, prediction, and Grad-CAM explainability | `app.py` (Streamlit Web Application) | **COMPLETE** |
| **2. Cleaned Metadata Dataset** | `dataset.csv` from project folder cleaned with logical anomalies resolved | `cleaned_dataset.csv` & `data/processed/cleaned_dataset.csv` | **COMPLETE** |
| **3. Original Raw Dataset** | Original 500 rows preserved without modification | `dataset.csv` & `data/raw/dataset.csv` | **COMPLETE** |
| **4. Completed Data Dictionary** | All 7 columns described with types, nullability, samples, and descriptions | `data_dictionary.csv` | **COMPLETE** |
| **5. EDA Script & Visualizations** | 3+ meaningful visualizations covering distributions, anomalies, and severity | `eda.py` & `reports/figures/*.svg` | **COMPLETE** |
| **6. Individual Project Report** | Minimum 8 pages covering Abstract, Literature Review, Methodology, Evaluation, Ethics | `Project_Report_HCLTech_Submission.md` | **COMPLETE** |
| **7. Project Synopsis Document** | Official template with 5W1H, DFD Level 0, Architecture, 3 Metrics, 2 Challenges | `Project_Synopsis_Submission.md` | **COMPLETE** |
| **8. Architecture Diagrams** | Usable diagram files for DFD Level 0 and System Architecture Block Diagram | `reports/figures/dfd_level_0.svg` & `system_architecture.svg` | **COMPLETE** |
| **9. Demo Script / Video Plan** | 1–2 minute video script or live presentation roadmap | `demo_script.md` | **COMPLETE** |
| **10. Ethical Compliance Declaration**| Signed declaration covering licensing, PII, DPDP Act 2023, and AI tool disclosure | Section 9 of Synopsis & Section 11 of Report | **COMPLETE** |
| **11. Automated Test Suite** | Unit tests for data pipeline, model logic, and Grad-CAM functionality | `tests/test_*.py` | **COMPLETE** |
| **12. Project Documentation & Setup** | Comprehensive README with execution instructions and setup commands | `README.md` | **COMPLETE** |

---

## REMAINING ACTIONS REQUIRED BY LEARNER BEFORE UPLOAD

1. **Fill in Your Personal Information:**
   - In `Project_Synopsis_Submission.md`: Replace `[FILL IN: Your Full Name]`, `[FILL IN: Roll Number]`, `[FILL IN: College Name]`, and `[FILL IN: Mentor Name]`.
   - In `Project_Report_HCLTech_Submission.md`: Replace placeholder header fields with your actual college and student details.

2. **Sign the Ethical Compliance Declaration:**
   - Section 9 of `Project_Synopsis_Submission.md` contains the signature line:  
     `Learner Signature: _________________________  Date: ________________`  
   - Print or digitally sign this section prior to final submission. *Do not leave blank.*

3. **Convert to Required Submission Formats:**
   - HCL Synopsis template specifies filename format:  
     `Project_Synopsis_[YourName]_[RollNo].docx` (or PDF).  
     *(You can open `Project_Synopsis_Submission.md` in Microsoft Word or Google Docs and save as `.docx` or `.pdf`)*.
   - Convert `Project_Report_HCLTech_Submission.md` to `Project_Report_[YourName]_[RollNo].pdf`.

4. **Record 1–2 Minute Demo Video (If Required by Campus Coordinator):**
   - Follow `demo_script.md` step-by-step.
   - Screen record the Streamlit web app in action (showing leaf image diagnosis and Grad-CAM heatmap).
   - Export video as MP4 (1080p or 720p).

5. **Zip the Final Submission Package:**
   - Ensure the zip contains all source code (`app.py`, `train_model.py`, `evaluate_model.py`, `data_preprocessing.py`, `project_config.py`, `gradcam.py`, `eda.py`), `cleaned_dataset.csv`, `data_dictionary.csv`, reports, and documentation.
