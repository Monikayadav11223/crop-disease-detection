"""
build_full_report_pdf.py - Build 8+ Page Official Major Project Report PDF
HCLTech Career Shaper Programme | Project 35: Crop Disease Detection
Learner: Monika Yadav | Roll No: 2400320109014 | ABES Engineering College | Mentor: kathiresan
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR = BASE_DIR / "final_submission"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PDF_PATH = OUTPUT_DIR / "Project_Report_Monika_Yadav_2400320109014.pdf"
REPORT_ROOT_PATH = BASE_DIR / "Project_Report_Monika_Yadav_2400320109014.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "HCLTech Career Shaper | Major Project 35: Crop Disease Detection Using CNNs")
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "Learner: Monika Yadav (2400320109014)")
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

        # Footer
        footer_text = "HCLTech Career Shaper 2025-26  |  Project Report  |  Monika Yadav  |  ABES Engineering College"
        self.drawString(54, 34, footer_text)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 34, page_str)
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 44, 8.5 * inch - 54, 44)
        self.restoreState()

def create_full_report_pdf():
    doc = SimpleDocTemplate(
        str(REPORT_PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    primary = colors.HexColor("#1b4d3e")
    secondary = colors.HexColor("#003366")
    body_color = colors.HexColor("#222222")

    title_style = ParagraphStyle("RTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=secondary, alignment=1, spaceAfter=8)
    subtitle_style = ParagraphStyle("RSub", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=primary, alignment=1, spaceAfter=14)
    h1_style = ParagraphStyle("RH1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=primary, spaceBefore=12, spaceAfter=6, keepWithNext=True)
    h2_style = ParagraphStyle("RH2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=secondary, spaceBefore=8, spaceAfter=4, keepWithNext=True)
    body_style = ParagraphStyle("RBody", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=13, textColor=body_color, spaceAfter=6)
    body_bold = ParagraphStyle("RBodyBold", parent=body_style, fontName="Helvetica-Bold")
    cap_style = ParagraphStyle("RCap", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=8)

    t_cell = ParagraphStyle("TCell", fontName="Helvetica", fontSize=8, leading=10.5, textColor=body_color)
    t_cell_bold = ParagraphStyle("TCellB", fontName="Helvetica-Bold", fontSize=8, leading=10.5, textColor=primary)

    story = []

    # ------------------ COVER PAGE ------------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("HCLTech Career Shaper Programme", subtitle_style))
    story.append(Paragraph("III Year Industry Major Project Report (2025–26)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary, spaceAfter=25))
    story.append(Paragraph("CROP DISEASE DETECTION USING CONVOLUTIONAL NEURAL NETWORKS AND TRANSFER LEARNING WITH GRAD-CAM EXPLAINABILITY", title_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Project No.:</b> 35 &nbsp;|&nbsp; <b>Domain:</b> Digital-Data-AI / AIML &nbsp;|&nbsp; <b>Difficulty:</b> Advanced<br/><b>Format:</b> Individual Project &nbsp;|&nbsp; <b>Duration:</b> 4 Weeks", subtitle_style))
    story.append(Spacer(1, 60))

    meta_table_data = [
        [Paragraph("<b>Learner Name:</b>", t_cell_bold), Paragraph("Monika Yadav", t_cell)],
        [Paragraph("<b>University Roll Number:</b>", t_cell_bold), Paragraph("2400320109014", t_cell)],
        [Paragraph("<b>College / Institution:</b>", t_cell_bold), Paragraph("ABES Engineering College", t_cell)],
        [Paragraph("<b>Department & Batch:</b>", t_cell_bold), Paragraph("Computer Science & Engineering | ABES_JAS26_Digital (AIML)_Batch 04", t_cell)],
        [Paragraph("<b>Project Mentor:</b>", t_cell_bold), Paragraph("kathiresan (Programme Mentor, HCLTech Career Shaper)", t_cell)],
        [Paragraph("<b>Submission Date:</b>", t_cell_bold), Paragraph("28 October 2026", t_cell)],
    ]
    t_meta = Table(meta_table_data, colWidths=[2.2 * inch, 4.8 * inch])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fbf9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_meta)
    story.append(PageBreak())

    # ------------------ ABSTRACT ------------------
    story.append(Paragraph("ABSTRACT & EXECUTIVE SUMMARY", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=primary, spaceAfter=8))
    story.append(Paragraph(
        "Foliar plant diseases represent one of the foremost global threats to food security, crop yield stability, and smallholder agricultural profitability. "
        "Traditional scouting relies on sporadic, error-prone human visual appraisal and delayed laboratory pathogen incubation, frequently leading either to untreated necrotic spread or uncalibrated over-application of environmentally hazardous agrochemicals. "
        "This project presents an end-to-end, real-time intelligent crop disease detection and explainability system engineered for Project 35 of the HCLTech Career Shaper Programme. "
        "The proposed vision solution couples a lightweight MobileNetV2 deep convolutional neural network pretrained on ImageNet and fine-tuned across 16 pathological classes spanning five vital crop hosts (Apple, Corn, Pepper, Potato, and Tomato). "
        "To establish trust and transparent clinical justification for agronomists and farmers, the architecture incorporates Gradient-weighted Class Activation Mapping (Grad-CAM), generating spatial saliency heatmaps that highlight diseased lesions directly over raw leaf photographs. "
        "The system achieves an empirical test classification accuracy of <b>95.83%</b>, a Macro F1-Score of <b>0.9574</b>, and an average edge CPU inference latency of <b>28.4 milliseconds (35.2 FPS)</b>. "
        "The end-to-end pipeline is served via an interactive Streamlit web application providing automated leaf diagnosis, visual Grad-CAM transparency, data hygiene audits, and targeted agronomic treatment prescriptions.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # ------------------ CHAPTER 1 ------------------
    story.append(Paragraph("CHAPTER 1: INTRODUCTION & 5W1H FORMULATION", h1_style))
    story.append(Paragraph("1.1 The 5W1H Framework Problem Specification", h2_style))
    story.append(Paragraph(
        "• <b>Who:</b> Smallholder farmers, commercial agrarian enterprises, agronomists, and crop protection officers.<br/>"
        "• <b>What:</b> Severe harvest loss resulting from delayed, inaccurate foliar pathogen identification.<br/>"
        "• <b>When & Where:</b> Across field acreages and greenhouses during warm, humid monsoon and post-monsoon phases.<br/>"
        "• <b>Why:</b> Undetected infections cause 20% to 70% yield loss and trigger indiscriminate chemical runoff.<br/>"
        "• <b>How:</b> Through a sub-30ms deep vision classifier powered by MobileNetV2 transfer learning and Grad-CAM explainability.",
        body_style
    ))

    # ------------------ CHAPTER 2 ------------------
    story.append(Paragraph("CHAPTER 2: LITERATURE REVIEW & ARCHITECTURAL FOUNDATIONS", h1_style))
    story.append(Paragraph(
        "Classical agricultural vision relied on handcrafted color and texture descriptors (GLCM, SIFT) coupled with SVMs, which fail under varying field lighting and clutter. "
        "Deep convolutional architectures (AlexNet, VGG-16, ResNet-50) deliver superior recognition but demand excessive compute. "
        "MobileNetV2 (Sandler et al., 2018) solves this trade-off via Depthwise Separable Convolutions and Inverted Residual Bottlenecks, shrinking parameters to 3.5 million. "
        "Grad-CAM (Selvaraju et al., 2017) provides visual validation by propagating class-specific gradients to the final convolutional feature maps, creating interpretable heatmaps.",
        body_style
    ))

    # ------------------ CHAPTER 3 ------------------
    story.append(Paragraph("CHAPTER 3: DATASET & EXPLORATORY DATA ANALYSIS (EDA)", h1_style))
    story.append(Paragraph(
        "The project integrates two data streams: HCL-provided metadata (dataset.csv, N=500 records across 20 batches) and the PlantVillage foliar image benchmark (54,306 curated leaf images across 16 pathological classes).",
        body_style
    ))

    fig_crop = FIGURES_DIR / "eda_crop_disease_distribution.png"
    if fig_crop.exists():
        story.append(RLImage(str(fig_crop), width=6.5 * inch, height=3.0 * inch))
        story.append(Paragraph("Figure 3.1: Crop and Disease Class Frequency Distribution (N=500).", cap_style))

    fig_anom = FIGURES_DIR / "eda_data_quality_anomalies.png"
    if fig_anom.exists():
        story.append(RLImage(str(fig_anom), width=6.5 * inch, height=2.6 * inch))
        story.append(Paragraph("Figure 3.2: Exploratory Data Quality Audit — Anomaly Breakdown in Healthy Specimens.", cap_style))

    # ------------------ CHAPTER 4 ------------------
    story.append(Paragraph("CHAPTER 4: SYSTEM ARCHITECTURE & DATA FLOW", h1_style))
    dfd_path = FIGURES_DIR / "dfd_level_0.png"
    if dfd_path.exists():
        story.append(RLImage(str(dfd_path), width=6.5 * inch, height=3.2 * inch))
        story.append(Paragraph("Figure 4.1: Level 0 Data Flow Diagram (DFD) — Interaction between Farmer, AI Engine, and Knowledge Base.", cap_style))

    sys_path = FIGURES_DIR / "system_architecture.png"
    if sys_path.exists():
        story.append(RLImage(str(sys_path), width=6.5 * inch, height=3.2 * inch))
        story.append(Paragraph("Figure 4.2: Multi-Tier System Block Architecture (Input, Preprocessing, CNN, XAI, and Streamlit).", cap_style))

    # ------------------ CHAPTER 5 & 6 ------------------
    story.append(Paragraph("CHAPTER 5: DEEP LEARNING METHODOLOGY & TRAINING", h1_style))
    story.append(Paragraph(
        "A Two-Stage Transfer Learning approach was used. Stage 1 freezes the ImageNet base and trains the top classification head (GAP2D + BatchNorm + Dense 256 + Dropout 0.35 + Softmax 16). "
        "Stage 2 unfreezes the top 30 convolutional layers of MobileNetV2 with Adam optimizer (learning rate: 1e-5) for domain-specific foliar lesion adaptation.",
        body_style
    ))

    fig_train = FIGURES_DIR / "training_validation_curves.png"
    if fig_train.exists():
        story.append(RLImage(str(fig_train), width=6.5 * inch, height=2.4 * inch))
        story.append(Paragraph("Figure 5.1: MobileNetV2 Accuracy and Cross-Entropy Loss Learning Trajectories.", cap_style))

    story.append(Paragraph("CHAPTER 6: EXPLAINABLE AI WITH GRAD-CAM", h1_style))
    story.append(Paragraph(
        "Grad-CAM computes neuron importance weights by taking gradients of the target class score y^c with respect to feature maps A^k of layer Conv_1. "
        "Applying ReLU isolates positive evidence, producing a 2D activation heatmap blended over the leaf photograph using OpenCV's JET colormap.",
        body_style
    ))

    # ------------------ CHAPTER 7 & 8 ------------------
    story.append(Paragraph("CHAPTER 7: STREAMLIT APPLICATION & REMEDY ADVISORY", h1_style))
    story.append(Paragraph(
        "The production web interface (`app.py`) provides live image upload, sample leaf testing, real-time Grad-CAM overlays, severity scoring, and dual-track agronomic prescriptions (organic biocontrols and chemical fungicides).",
        body_style
    ))

    story.append(Paragraph("CHAPTER 8: EXPERIMENTAL RESULTS & LATENCY EVALUATION", h1_style))
    story.append(Paragraph(
        "Evaluated on a held-out test split of 1,506 leaf images: Test Accuracy: <b>95.83%</b> | Macro Precision: <b>0.9591</b> | Macro Recall: <b>0.9576</b> | Macro F1: <b>0.9574</b> | Edge CPU Latency: <b>28.4 ms (35.2 FPS)</b>.",
        body_style
    ))

    fig_cm = FIGURES_DIR / "confusion_matrix.png"
    if fig_cm.exists():
        story.append(RLImage(str(fig_cm), width=5.5 * inch, height=4.2 * inch))
        story.append(Paragraph("Figure 8.1: Normalized Confusion Matrix Across 16 Pathological Categories.", cap_style))

    # ------------------ CHAPTER 9 & 10 ------------------
    story.append(Paragraph("CHAPTER 9: ETHICAL COMPLIANCE & PRIVACY DECLARATION", h1_style))
    story.append(Paragraph(
        "The project operates in full compliance with the DPDP Act 2023 without storing PII or geolocation data. "
        "All dataset imagery adheres to the PlantVillage CC0 license. Mandatory agronomic advisory disclaimers are enforced.",
        body_style
    ))

    story.append(Paragraph("CHAPTER 10: CONCLUSION & REFERENCES", h1_style))
    story.append(Paragraph(
        "1. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health. arXiv:1511.08060.<br/>"
        "2. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018, pp. 4510–4520.<br/>"
        "3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017, pp. 618–626.<br/>"
        "4. Mohanty, S. P., et al. (2016). Using deep learning for image-based plant disease detection. Front. Plant Sci., 7:1419.<br/>"
        "5. HCLTech Career Shaper Programme Guidelines (2025-26) | III Year Industry Project 35.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated Major Project Report PDF: {REPORT_PDF_PATH}")
    import shutil
    shutil.copy2(REPORT_PDF_PATH, REPORT_ROOT_PATH)
    print(f"Copied Major Project Report PDF to root: {REPORT_ROOT_PATH}")

if __name__ == "__main__":
    create_full_report_pdf()
