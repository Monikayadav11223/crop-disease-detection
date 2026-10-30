# 2-Minute Project Demonstration Video Script (Official English Version)
### HCLTech Career Shaper Programme | Industry Project 35
**Project Title:** Crop Disease Detection Using Convolutional Neural Networks  
**Learner Name:** Monika Yadav  
**Roll Number:** 2400320109014  
**College:** ABES Engineering College  
**Batch:** ABES_JAS26_Digital (AIML)_Batch 04  
**Mentor:** Mr. kathiresan  
**Target Duration:** 90 – 120 Seconds (1.5 to 2.0 Minutes)  
**Application URL:** http://localhost:8501  

---

## 🎬 VIDEO TIMELINE & SCREENPLAY

| Timestamp | Screen / Visual Action | Key Talking Points |
| :--- | :--- | :--- |
| **00:00 – 00:25** | Home Dashboard (`http://localhost:8501`) | Introduction, Personal Details, Problem Statement |
| **00:25 – 00:55** | **Tab 1: 🌿 Live Foliar Diagnosis** | Live inference demo, Grad-CAM visual heatmap, IPM prescriptions |
| **00:55 – 01:20** | **Tab 2: 📊 HCL Metadata & Quality Audit** | 500-row telemetry audit, anomaly resolution, risk assessment |
| **01:20 – 01:45** | **Tab 3: 📈 CNN Performance & Metrics** | 95.83% accuracy, confusion matrix, 28.4ms CPU latency |
| **01:45 – 02:00** | **Tab 4: 📋 Synopsis & Ethics** | DPDP Act 2023 compliance, conclusion & acknowledgment |

---

## 🎙️ WORD-FOR-WORD SPOKEN SCRIPT

### [00:00 – 00:25] 1. Introduction & Agricultural Problem Formulation
*(Visual: Show the home screen of the application in browser with title banner)*

> "Good morning, esteemed mentors and evaluators.  
> My name is **Monika Yadav**, University Roll Number **2400320109014**, from **ABES Engineering College**. I am completing my HCLTech Career Shaper III-Year Industry Project under the mentorship of Mr. **kathiresan**.  
> 
> Today, I am demonstrating Project Number 35: **Crop Disease Detection Using Convolutional Neural Networks with Grad-CAM Explainability**.  
> 
> Across global agriculture, foliar fungal and bacterial pathogens destroy between 20% and 40% of prospective crop yields each year. Traditional scouting depends on delayed, error-prone visual checks, leading to severe harvest loss and excessive chemical pesticide spraying. To solve this, I built an automated, real-time edge vision diagnostic platform."

---

### [00:25 – 00:55] 2. Live Foliar Diagnosis & Grad-CAM Explainability
*(Visual: Click on **Tab 1: 🌿 Live Foliar Diagnosis & Grad-CAM**. In the sample selector, choose **'Tomato Early Blight'**)*

> "Here in our Live Diagnostic Studio, a user can upload a leaf photograph or select from test specimens. When I select this **Tomato Leaf**, our **MobileNetV2 Transfer Learning** model classifies it in just **28 milliseconds** as **Tomato - Early Blight** with a **96.2% confidence score**.  
> 
> Crucially, our system eliminates black-box opacity through **Gradient-weighted Class Activation Mapping (Grad-CAM)**. As highlighted by the red-and-yellow activation heatmap on screen, the network specifically focuses on the characteristic target-board necrotic lesions and chlorotic halos.  
> 
> Furthermore, the application immediately delivers integrated agronomic prescriptions—offering both biological bio-fungicide remedies and targeted chemical treatment thresholds."

---

### [00:55 – 01:20] 3. HCL Metadata Telemetry & Data Hygiene Audit
*(Visual: Click on **Tab 2: 📊 HCL Metadata & Quality Audit**. Scroll down to show the anomaly breakdown bar chart)*

> "Next, addressing Phase 2 of our project requirements: we audited the official 500-record field telemetry dataset (`dataset.csv`).  
> 
> Our exploratory audit uncovered a significant domain contradiction where 69 records classified as 'Healthy' carried conflicting infection severity tags.  
> 
> We engineered an automated, non-destructive data cleaning pipeline that harmonized healthy severity ratings to 'None', flagged recording anomalies for full traceability, and computed multi-tier risk levels across all 20 collection batches."

---

### [01:20 – 01:45] 4. Deep Learning Model Performance & Latency Benchmarks
*(Visual: Click on **Tab 3: 📈 CNN Performance & Metrics**. Point to the 16-class Confusion Matrix and Training Curves)*

> "Turning to empirical validation: evaluated across a held-out test split of 1,506 leaf images, our model achieves:
> • An **Overall Test Accuracy of 95.83%**,  
> • A **Balanced Macro F1-Score of 0.9574**, and  
> • An average inference latency of just **28.4 milliseconds** per image—achieving over **35 frames per second** on standard Intel CPU architecture without requiring expensive GPU hardware."

---

### [01:45 – 02:00] 5. Ethical Compliance & Conclusion
*(Visual: Click on **Tab 4: 📋 Project Synopsis & Ethics**. Show the compliance checkboxes and signature block)*

> "Finally, our project complies fully with the **Digital Personal Data Protection (DPDP) Act 2023**, processing zero PII, and operates under open-source academic licenses with mandatory agronomic advisory disclaimers.  
> 
> Thank you, mentors and HCLTech evaluators. I am now ready for your questions."

---

## 💡 Quick Tips for Flawless Recording
1. **Screen Resolution:** Keep your browser maximized at 100% or 110% zoom for crisp, readable text.
2. **Recording Shortcut:** Press `Win + Alt + R` on Windows to record your screen and microphone instantly.
3. **Pacing:** Speak at a steady, calm, confident pace. 180 to 220 words per minute is ideal for a 2-minute video.
