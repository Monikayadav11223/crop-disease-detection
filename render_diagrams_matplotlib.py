"""
render_diagrams_matplotlib.py - Generate high-resolution PNGs of DFD Level 0 and System Architecture using Matplotlib
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def render_dfd_level_0():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header Banner
    banner = patches.FancyBboxPatch((2, 88), 96, 10, boxstyle="round,pad=0.5", facecolor="#1b4d3e", edgecolor="none")
    ax.add_patch(banner)
    ax.text(50, 93, "Data Flow Diagram (DFD Level 0) - Crop Disease Detection System", color="white", fontsize=13, weight="bold", ha="center", va="center")

    # External Entity: Farmer
    farmer_box = patches.FancyBboxPatch((5, 38), 20, 24, boxstyle="round,pad=0.5", facecolor="#34495e", edgecolor="#2c3e50", linewidth=1.5)
    ax.add_patch(farmer_box)
    ax.text(15, 54, "External Entity", color="#ecf0f1", fontsize=9, ha="center")
    ax.text(15, 49, "Farmer / Agronomist", color="white", fontsize=10, weight="bold", ha="center")
    ax.text(15, 43, "(End User)", color="#bdc3c7", fontsize=8, ha="center")

    # Central Process 0.0
    circle = patches.Circle((50, 50), 16, facecolor="#e8f5e9", edgecolor="#27ae60", linewidth=2.5)
    ax.add_patch(circle)
    ax.text(50, 57, "Process 0.0", color="#1b4d3e", fontsize=10, weight="bold", ha="center")
    ax.text(50, 51.5, "Crop Disease Classification", color="#1b4d3e", fontsize=9.5, weight="bold", ha="center")
    ax.text(50, 46.5, "& Explainability Engine", color="#1b4d3e", fontsize=9.5, weight="bold", ha="center")
    ax.text(50, 40.5, "(MobileNetV2 + Grad-CAM)", color="#27ae60", fontsize=8.5, weight="bold", ha="center")

    # Data Store D1: Model
    d1_box = patches.FancyBboxPatch((75, 62), 22, 18, boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#2980b9", linewidth=1.5)
    ax.add_patch(d1_box)
    ax.plot([77, 77], [62, 80], color="#2980b9", linewidth=1.5)
    ax.text(86, 75, "D1: Trained CNN Model", color="#2980b9", fontsize=9, weight="bold", ha="center")
    ax.text(86, 69, "MobileNetV2 Weights & Registry", color="#555555", fontsize=7.5, ha="center")

    # Data Store D2: Database
    d2_box = patches.FancyBboxPatch((75, 20), 22, 18, boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#8e44ad", linewidth=1.5)
    ax.add_patch(d2_box)
    ax.plot([77, 77], [20, 38], color="#8e44ad", linewidth=1.5)
    ax.text(86, 33, "D2: Agronomic Database", color="#8e44ad", fontsize=9, weight="bold", ha="center")
    ax.text(86, 27, "Cleaned dataset.csv & Protocols", color="#555555", fontsize=7.5, ha="center")

    # Arrows
    # Farmer -> Process
    ax.annotate("", xy=(34, 53), xytext=(25, 53), arrowprops=dict(arrowstyle="->", color="#1b4d3e", lw=2))
    ax.text(29.5, 56, "1. Upload Leaf Image", color="#1b4d3e", fontsize=7.5, weight="bold", ha="center")

    # Process -> Farmer
    ax.annotate("", xy=(25, 45), xytext=(34, 45), arrowprops=dict(arrowstyle="->", color="#1b4d3e", lw=2))
    ax.text(29.5, 40.5, "4. Prediction & Heatmap", color="#1b4d3e", fontsize=7.5, weight="bold", ha="center")

    # Process -> D1
    ax.annotate("", xy=(75, 71), xytext=(63, 60), arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.8))
    ax.text(66, 68, "2. Forward Pass", color="#2980b9", fontsize=7.5, ha="center")

    # D2 -> Process
    ax.annotate("", xy=(63, 40), xytext=(75, 29), arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.8))
    ax.text(66, 31, "3. Protocols", color="#8e44ad", fontsize=7.5, ha="center")

    # Footer
    ax.text(50, 5, "Project No. 35 | HCLTech Career Shaper 2025-26 | Gane & Sarson DFD Level 0 Standard", color="#7f8c8d", fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dfd_level_0.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated dfd_level_0.png via Matplotlib")


def render_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title Banner
    banner = patches.FancyBboxPatch((2, 90), 96, 8, boxstyle="round,pad=0.5", facecolor="#145A32", edgecolor="none")
    ax.add_patch(banner)
    ax.text(50, 94, "System Block Architecture - Crop Disease Detection Engine", color="white", fontsize=14, weight="bold", ha="center", va="center")

    # Column 1: Input Tier
    c1 = patches.FancyBboxPatch((3, 10), 21, 76, boxstyle="round,pad=0.4", facecolor="white", edgecolor="#2980b9", linewidth=1.5)
    ax.add_patch(c1)
    b1 = patches.FancyBboxPatch((3, 80), 21, 6, boxstyle="round,pad=0.2", facecolor="#2980b9", edgecolor="none")
    ax.add_patch(b1)
    ax.text(13.5, 83, "1. Input Layer", color="white", fontsize=10, weight="bold", ha="center")

    sub1 = patches.FancyBboxPatch((4.5, 58), 18, 19, boxstyle="round,pad=0.3", facecolor="#ebf5fb", edgecolor="#aed6f1")
    ax.add_patch(sub1)
    ax.text(13.5, 73, "Leaf Image Input", color="#1b4f72", fontsize=9, weight="bold", ha="center")
    ax.text(13.5, 67, "- PlantVillage Images\n- User Upload (JPG/PNG)", color="#444444", fontsize=7.5, ha="center")

    sub2 = patches.FancyBboxPatch((4.5, 36), 18, 19, boxstyle="round,pad=0.3", facecolor="#ebf5fb", edgecolor="#aed6f1")
    ax.add_patch(sub2)
    ax.text(13.5, 51, "HCL Metadata", color="#1b4f72", fontsize=9, weight="bold", ha="center")
    ax.text(13.5, 44, "- dataset.csv (N=500)\n- 5 Crops / 5 Classes\n- 20 Field Batches", color="#444444", fontsize=7.5, ha="center")

    sub3 = patches.FancyBboxPatch((4.5, 14), 18, 19, boxstyle="round,pad=0.3", facecolor="#ebf5fb", edgecolor="#aed6f1")
    ax.add_patch(sub3)
    ax.text(13.5, 29, "Agronomic KB", color="#1b4f72", fontsize=9, weight="bold", ha="center")
    ax.text(13.5, 22, "- Pathogen Taxonomy\n- Organic Protocols\n- Chemical Prescriptions", color="#444444", fontsize=7.5, ha="center")

    # Column 2: Preprocessing Tier
    c2 = patches.FancyBboxPatch((27, 10), 21, 76, boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e67e22", linewidth=1.5)
    ax.add_patch(c2)
    b2 = patches.FancyBboxPatch((27, 80), 21, 6, boxstyle="round,pad=0.2", facecolor="#e67e22", edgecolor="none")
    ax.add_patch(b2)
    ax.text(37.5, 83, "2. Preprocessing", color="white", fontsize=10, weight="bold", ha="center")

    sub2_1 = patches.FancyBboxPatch((28.5, 58), 18, 19, boxstyle="round,pad=0.3", facecolor="#fef5e7", edgecolor="#f8c471")
    ax.add_patch(sub2_1)
    ax.text(37.5, 73, "Image Pipeline", color="#b9770e", fontsize=9, weight="bold", ha="center")
    ax.text(37.5, 67, "- Resize 224x224 px\n- Normalization [-1, 1]\n- RGB 3-Channels", color="#444444", fontsize=7.5, ha="center")

    sub2_2 = patches.FancyBboxPatch((28.5, 36), 18, 19, boxstyle="round,pad=0.3", facecolor="#fef5e7", edgecolor="#f8c471")
    ax.add_patch(sub2_2)
    ax.text(37.5, 51, "Data Hygiene", color="#b9770e", fontsize=9, weight="bold", ha="center")
    ax.text(37.5, 44, "- Healthy Severity = None\n- Anomaly Audit Flags\n- Imputed Risk Index", color="#444444", fontsize=7.5, ha="center")

    sub2_3 = patches.FancyBboxPatch((28.5, 14), 18, 19, boxstyle="round,pad=0.3", facecolor="#fef5e7", edgecolor="#f8c471")
    ax.add_patch(sub2_3)
    ax.text(37.5, 29, "Augmentation", color="#b9770e", fontsize=9, weight="bold", ha="center")
    ax.text(37.5, 22, "- Random Flips (H/V)\n- Rotation & Zoom\n- Stratified Splits (70/15/15)", color="#444444", fontsize=7.5, ha="center")

    # Column 3: Deep Learning Tier
    c3 = patches.FancyBboxPatch((51, 10), 22, 76, boxstyle="round,pad=0.4", facecolor="white", edgecolor="#27ae60", linewidth=1.5)
    ax.add_patch(c3)
    b3 = patches.FancyBboxPatch((51, 80), 22, 6, boxstyle="round,pad=0.2", facecolor="#27ae60", edgecolor="none")
    ax.add_patch(b3)
    ax.text(62, 83, "3. Deep Learning Core", color="white", fontsize=10, weight="bold", ha="center")

    sub3_1 = patches.FancyBboxPatch((52.5, 46), 19, 31, boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#a9dfbf")
    ax.add_patch(sub3_1)
    ax.text(62, 73, "MobileNetV2 CNN", color="#196f3d", fontsize=9, weight="bold", ha="center")
    ax.text(62, 63, "- Inverted Residuals\n- Depthwise Conv\n- GlobalAvgPool2D\n- Dropout (0.35)\n- Softmax (16 Classes)\n* Accuracy: 95.83%\n* Latency: 28.4 ms (CPU)", color="#333333", fontsize=7.2, ha="center")

    sub3_2 = patches.FancyBboxPatch((52.5, 14), 19, 29, boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#a9dfbf")
    ax.add_patch(sub3_2)
    ax.text(62, 39, "Grad-CAM XAI", color="#196f3d", fontsize=9, weight="bold", ha="center")
    ax.text(62, 28, "- Last Conv: Conv_1\n- Class Score Gradients\n- ReLU Saliency Map\n- OpenCV JET Overlay\n* Verifies Lesion Focus", color="#333333", fontsize=7.2, ha="center")

    # Column 4: Application Tier
    c4 = patches.FancyBboxPatch((76, 10), 21, 76, boxstyle="round,pad=0.4", facecolor="white", edgecolor="#8e44ad", linewidth=1.5)
    ax.add_patch(c4)
    b4 = patches.FancyBboxPatch((76, 80), 21, 6, boxstyle="round,pad=0.2", facecolor="#8e44ad", edgecolor="none")
    ax.add_patch(b4)
    ax.text(86.5, 83, "4. Streamlit Portal", color="white", fontsize=10, weight="bold", ha="center")

    sub4 = patches.FancyBboxPatch((77.5, 14), 18, 63, boxstyle="round,pad=0.3", facecolor="#f4ecf7", edgecolor="#d2b4de")
    ax.add_patch(sub4)
    ax.text(86.5, 73, "Interactive UI", color="#5b2c6f", fontsize=9, weight="bold", ha="center")
    ax.text(86.5, 52, "Tab 1: Live Diagnosis\n- Image Upload\n- Class & Confidence\n- Grad-CAM Heatmap\n- Organic/Chem Advisory\n\nTab 2: HCL Analytics\n- Raw vs Clean Data\n- Batch Slicing\n\nTab 3: Model Metrics\n- Confusion Matrix\n- Accuracy Trajectories\n\nTab 4: Ethics & Synopsis\n- DPDP Act 2023", color="#333333", fontsize=7.2, ha="center")

    # Connectors
    ax.annotate("", xy=(27, 48), xytext=(24, 48), arrowprops=dict(arrowstyle="->", color="#145A32", lw=2.5))
    ax.annotate("", xy=(51, 48), xytext=(48, 48), arrowprops=dict(arrowstyle="->", color="#145A32", lw=2.5))
    ax.annotate("", xy=(76, 48), xytext=(73, 48), arrowprops=dict(arrowstyle="->", color="#145A32", lw=2.5))

    ax.text(50, 4, "HCLTech Career Shaper 2025-26 | Project No. 35 | End-to-End Deep Learning Architecture", color="#7f8c8d", fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "system_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated system_architecture.png via Matplotlib")

if __name__ == "__main__":
    render_dfd_level_0()
    render_system_architecture()
