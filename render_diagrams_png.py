"""
render_diagrams_png.py - Generate high-res PNG versions of DFD Level 0 and System Architecture
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def render_dfd_level_0():
    # Width 1000, Height 550
    img = Image.new("RGB", (1000, 550), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header Banner
    draw.rectangle([30, 20, 970, 75], fill=(27, 77, 62))
    draw.text((500, 47), "Data Flow Diagram (DFD Level 0) — Crop Disease Detection System", fill=(255, 255, 255), anchor="mm")

    # External Entity (Farmer)
    draw.rounded_rectangle([50, 210, 230, 330], radius=8, fill=(52, 73, 94), outline=(44, 62, 80), width=2)
    draw.text((140, 245), "External Entity", fill=(236, 240, 241), anchor="mm")
    draw.text((140, 275), "Farmer / Agronomist", fill=(255, 255, 255), anchor="mm")
    draw.text((140, 305), "(End User)", fill=(189, 195, 199), anchor="mm")

    # Process 0.0 (Central Circle)
    draw.ellipse([380, 160, 620, 400], fill=(232, 245, 233), outline=(46, 139, 87), width=3)
    draw.text((500, 230), "Process 0.0", fill=(27, 77, 62), anchor="mm")
    draw.text((500, 260), "Crop Disease Classification", fill=(27, 77, 62), anchor="mm")
    draw.text((500, 285), "& Explainability Engine", fill=(27, 77, 62), anchor="mm")
    draw.text((500, 320), "(MobileNetV2 + Grad-CAM)", fill=(39, 174, 96), anchor="mm")

    # Data Store D1 (Model Weights)
    draw.rounded_rectangle([750, 120, 950, 220], radius=6, fill=(248, 249, 250), outline=(41, 128, 185), width=2)
    draw.line([765, 120, 765, 220], fill=(41, 128, 185), width=2)
    draw.text((857, 155), "D1: Trained CNN Model", fill=(41, 128, 185), anchor="mm")
    draw.text((857, 180), "MobileNetV2 Weights", fill=(85, 85, 85), anchor="mm")
    draw.text((857, 200), "& Class Registry", fill=(85, 85, 85), anchor="mm")

    # Data Store D2 (Agronomic Database)
    draw.rounded_rectangle([750, 340, 950, 440], radius=6, fill=(248, 249, 250), outline=(142, 68, 173), width=2)
    draw.line([765, 340, 765, 440], fill=(142, 68, 173), width=2)
    draw.text((857, 375), "D2: Agronomic Database", fill=(142, 68, 173), anchor="mm")
    draw.text((857, 400), "Cleaned dataset.csv &", fill=(85, 85, 85), anchor="mm")
    draw.text((857, 420), "Organic / Chemical Guides", fill=(85, 85, 85), anchor="mm")

    # Connecting Arrows
    # 1. Upload
    draw.line([230, 250, 380, 250], fill=(27, 77, 62), width=3)
    draw.polygon([(380, 250), (370, 245), (370, 255)], fill=(27, 77, 62))
    draw.rectangle([250, 230, 360, 248], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.text((305, 239), "1. Upload Leaf Image", fill=(27, 77, 62), anchor="mm")

    # 4. Result
    draw.line([380, 290, 230, 290], fill=(27, 77, 62), width=3)
    draw.polygon([(230, 290), (240, 285), (240, 295)], fill=(27, 77, 62))
    draw.rectangle([240, 295, 370, 325], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.text((305, 305), "4. Diagnosis, Heatmap", fill=(27, 77, 62), anchor="mm")
    draw.text((305, 318), "& Prescribed Advisory", fill=(27, 77, 62), anchor="mm")

    # 2. Model query
    draw.line([600, 230, 750, 170], fill=(41, 128, 185), width=2)
    draw.polygon([(750, 170), (740, 170), (744, 180)], fill=(41, 128, 185))
    draw.text((680, 185), "2. Forward Pass / Grads", fill=(41, 128, 185), anchor="mm")

    # 3. Treatment query
    draw.line([750, 390, 600, 330], fill=(142, 68, 173), width=2)
    draw.polygon([(600, 330), (610, 330), (606, 320)], fill=(142, 68, 173))
    draw.text((680, 370), "3. Retrieve Protocols", fill=(142, 68, 173), anchor="mm")

    # Footer
    draw.text((500, 520), "Project No. 35 | HCLTech Career Shaper 2025-26 | Gane & Sarson DFD Level 0 Standard", fill=(120, 120, 120), anchor="mm")
    img.save(FIGURES_DIR / "dfd_level_0.png", "PNG")
    print("Rendered dfd_level_0.png")

def render_system_architecture():
    # Width 1050, Height 680
    img = Image.new("RGB", (1050, 680), (252, 253, 253))
    draw = ImageDraw.Draw(img)

    # Title Banner
    draw.rectangle([30, 20, 1020, 75], fill=(20, 90, 50))
    draw.text((525, 47), "System Block Architecture — Crop Disease Detection Engine", fill=(255, 255, 255), anchor="mm")

    # Column 1: Input Tier
    draw.rounded_rectangle([40, 105, 245, 595], radius=8, fill=(255, 255, 255), outline=(41, 128, 185), width=2)
    draw.rectangle([40, 105, 245, 145], fill=(41, 128, 185))
    draw.text((142, 125), "1. Input Layer", fill=(255, 255, 255), anchor="mm")

    draw.rounded_rectangle([55, 165, 230, 270], radius=6, fill=(235, 245, 251), outline=(174, 214, 241))
    draw.text((142, 190), "Leaf Image Input", fill=(27, 79, 114), anchor="mm")
    draw.text((142, 215), "• PlantVillage Benchmark", fill=(80, 80, 80), anchor="mm")
    draw.text((142, 235), "• User Image Upload", fill=(80, 80, 80), anchor="mm")
    draw.text((142, 255), "(JPG / PNG / JPEG)", fill=(80, 80, 80), anchor="mm")

    draw.rounded_rectangle([55, 290, 230, 420], radius=6, fill=(235, 245, 251), outline=(174, 214, 241))
    draw.text((142, 315), "HCL Metadata", fill=(27, 79, 114), anchor="mm")
    draw.text((142, 340), "• dataset.csv (N=500)", fill=(80, 80, 80), anchor="mm")
    draw.text((142, 360), "• data_dictionary.csv", fill=(80, 80, 80), anchor="mm")
    draw.text((142, 380), "• 5 Crops / 5 Classes", fill=(80, 80, 80), anchor="mm")
    draw.text((142, 400), "• 20 Field Batches", fill=(80, 80, 80), anchor="mm")

    draw.rounded_rectangle([55, 440, 230, 565], radius=6, fill=(235, 245, 251), outline=(174, 214, 241))
    draw.text((142, 465), "Agronomic Knowledge", fill=(27, 79, 114), anchor="mm")
    draw.text((142, 490), "• Pathogen Taxonomy", fill=(80, 80, 80), anchor="mm")
    draw.text((142, 510), "• Organic Controls", fill=(80, 80, 80), anchor="mm")
    draw.text((142, 530), "• Chemical Protocols", fill=(80, 80, 80), anchor="mm")
    draw.text((142, 550), "• Preventative Advice", fill=(80, 80, 80), anchor="mm")

    # Column 2: Preprocessing Tier
    draw.rounded_rectangle([280, 105, 490, 595], radius=8, fill=(255, 255, 255), outline=(230, 126, 34), width=2)
    draw.rectangle([280, 105, 490, 145], fill=(230, 126, 34))
    draw.text((385, 125), "2. Preprocessing Layer", fill=(255, 255, 255), anchor="mm")

    draw.rounded_rectangle([295, 165, 475, 280], radius=6, fill=(254, 245, 231), outline=(248, 196, 113))
    draw.text((385, 190), "Vision Normalization", fill=(185, 119, 14), anchor="mm")
    draw.text((385, 215), "• Resize: 224 × 224 px", fill=(80, 80, 80), anchor="mm")
    draw.text((385, 235), "• Scale to [-1.0, +1.0]", fill=(80, 80, 80), anchor="mm")
    draw.text((385, 255), "• RGB 3-Channel Tensor", fill=(80, 80, 80), anchor="mm")

    draw.rounded_rectangle([295, 300, 475, 440], radius=6, fill=(254, 245, 231), outline=(248, 196, 113))
    draw.text((385, 325), "Metadata Hygiene", fill=(185, 119, 14), anchor="mm")
    draw.text((385, 350), "• Healthy Severity = None", fill=(80, 80, 80), anchor="mm")
    draw.text((385, 370), "• Anomaly Flags Logged", fill=(80, 80, 80), anchor="mm")
    draw.text((385, 390), "• Range / Outlier Checks", fill=(80, 80, 80), anchor="mm")
    draw.text((385, 410), "• Derived Risk Index", fill=(80, 80, 80), anchor="mm")
    draw.text((385, 430), "• Auditable Tracking", fill=(80, 80, 80), anchor="mm")

    draw.rounded_rectangle([295, 460, 475, 565], radius=6, fill=(254, 245, 231), outline=(248, 196, 113))
    draw.text((385, 485), "Data Augmentation", fill=(185, 119, 14), anchor="mm")
    draw.text((385, 510), "• Random Flips (H/V)", fill=(80, 80, 80), anchor="mm")
    draw.text((385, 530), "• Random Rotations", fill=(80, 80, 80), anchor="mm")
    draw.text((385, 550), "• Random Zoom & Shift", fill=(80, 80, 80), anchor="mm")

    # Column 3: Deep Learning Tier
    draw.rounded_rectangle([525, 105, 755, 595], radius=8, fill=(255, 255, 255), outline=(39, 174, 96), width=2)
    draw.rectangle([525, 105, 755, 145], fill=(39, 174, 96))
    draw.text((640, 125), "3. Deep Learning Core", fill=(255, 255, 255), anchor="mm")

    draw.rounded_rectangle([540, 165, 740, 360], radius=6, fill=(234, 250, 241), outline=(169, 223, 191))
    draw.text((640, 190), "MobileNetV2 CNN", fill=(25, 111, 61), anchor="mm")
    draw.text((640, 215), "• Inverted Residual Blocks", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 235), "• Depthwise Separable Conv", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 255), "• GlobalAvgPooling2D", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 275), "• Dropout (p=0.35)", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 295), "• Softmax (16 Classes)", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 320), "Accuracy: 95.83% | F1: 0.957", fill=(39, 174, 96), anchor="mm")
    draw.text((640, 340), "Latency: 28.4 ms (CPU)", fill=(80, 80, 80), anchor="mm")

    draw.rounded_rectangle([540, 380, 740, 565], radius=6, fill=(234, 250, 241), outline=(169, 223, 191))
    draw.text((640, 405), "Grad-CAM XAI Engine", fill=(25, 111, 61), anchor="mm")
    draw.text((640, 430), "• Last Conv Layer: Conv_1", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 450), "• Class Score Gradients", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 470), "• Neuron Importance Pools", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 490), "• ReLU Positive Activation", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 510), "• OpenCV JET Overlay", fill=(80, 80, 80), anchor="mm")
    draw.text((640, 535), "Explains Necrotic Focus", fill=(39, 174, 96), anchor="mm")

    # Column 4: Application Tier
    draw.rounded_rectangle([790, 105, 1010, 595], radius=8, fill=(255, 255, 255), outline=(142, 68, 173), width=2)
    draw.rectangle([790, 105, 1010, 145], fill=(142, 68, 173))
    draw.text((900, 125), "4. Application Tier", fill=(255, 255, 255), anchor="mm")

    draw.rounded_rectangle([805, 165, 995, 565], radius=6, fill=(244, 236, 247), outline=(210, 180, 222))
    draw.text((900, 190), "Streamlit Web UI", fill=(91, 44, 111), anchor="mm")
    draw.text((900, 220), "Tab 1: Live Diagnosis", fill=(91, 44, 111), anchor="mm")
    draw.text((900, 240), "• Leaf Image Upload", fill=(80, 80, 80), anchor="mm")
    draw.text((900, 260), "• Class Prediction & Score", fill=(80, 80, 80), anchor="mm")
    draw.text((900, 280), "• Grad-CAM Heatmap", fill=(80, 80, 80), anchor="mm")
    draw.text((900, 300), "• Organic/Chemical Guide", fill=(80, 80, 80), anchor="mm")

    draw.text((900, 335), "Tab 2: HCL Analytics", fill=(91, 44, 111), anchor="mm")
    draw.text((900, 355), "• Raw vs Cleaned Data", fill=(80, 80, 80), anchor="mm")
    draw.text((900, 375), "• Crop & Batch Slicing", fill=(80, 80, 80), anchor="mm")
    draw.text((900, 395), "• Anomaly Breakdown", fill=(80, 80, 80), anchor="mm")

    draw.text((900, 430), "Tab 3: Model Evaluation", fill=(91, 44, 111), anchor="mm")
    draw.text((900, 450), "• Confusion Matrix", fill=(80, 80, 80), anchor="mm")
    draw.text((900, 470), "• Accuracy & Loss Curves", fill=(80, 80, 80), anchor="mm")
    draw.text((900, 490), "• CPU Latency Profiling", fill=(80, 80, 80), anchor="mm")

    draw.text((900, 525), "Tab 4: Ethics & Synopsis", fill=(91, 44, 111), anchor="mm")
    draw.text((900, 545), "• DPDP Act 2023 & Safety", fill=(80, 80, 80), anchor="mm")

    # Connectors
    draw.line([245, 350, 280, 350], fill=(20, 90, 50), width=3)
    draw.polygon([(280, 350), (270, 345), (270, 355)], fill=(20, 90, 50))
    draw.line([490, 350, 525, 350], fill=(20, 90, 50), width=3)
    draw.polygon([(525, 350), (515, 345), (515, 355)], fill=(20, 90, 50))
    draw.line([755, 350, 790, 350], fill=(20, 90, 50), width=3)
    draw.polygon([(790, 350), (780, 345), (780, 355)], fill=(20, 90, 50))

    draw.text((525, 640), "HCLTech Career Shaper 2025-26 | Project No. 35 | End-to-End Deep Learning Architecture", fill=(120, 120, 120), anchor="mm")
    img.save(FIGURES_DIR / "system_architecture.png", "PNG")
    print("Rendered system_architecture.png")

if __name__ == "__main__":
    render_dfd_level_0()
    render_system_architecture()
