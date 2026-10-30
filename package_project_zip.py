"""
package_project_zip.py - Package and verify the complete HCLTech submission ZIP for Monika Yadav
"""

import os
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = BASE_DIR / "final_submission"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

ZIP_NAME = "Crop_Disease_Detection_Monika_Yadav_2400320109014.zip"
ZIP_OUTPUT_PATH = SUBMISSION_DIR / ZIP_NAME
ROOT_ZIP_OUTPUT_PATH = BASE_DIR / ZIP_NAME

LEGACY_ZIP_NAME = "Crop_Disease_Detection_HCL.zip"
LEGACY_ZIP_PATH = SUBMISSION_DIR / LEGACY_ZIP_NAME
ROOT_LEGACY_ZIP_PATH = BASE_DIR / LEGACY_ZIP_NAME

EXCLUDE_DIRS = {
    "venv", ".git", "__pycache__", "temp_mpl", "final_submission",
    ".pytest_cache", ".system_generated", ".keras"
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".tmp"}


def create_submission_zip():
    print(f"Packaging project from {BASE_DIR}...")
    file_count = 0

    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(BASE_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in EXCLUDE_EXTS or file.startswith(".") or file.endswith(".zip"):
                    continue

                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(BASE_DIR)

                archive_name = Path("crop_disease_detection") / rel_path
                zipf.write(abs_path, arcname=str(archive_name))
                file_count += 1

    print(f"Added {file_count} files to {ZIP_OUTPUT_PATH}")

    # Copy to root and legacy filenames
    shutil.copy2(ZIP_OUTPUT_PATH, ROOT_ZIP_OUTPUT_PATH)
    shutil.copy2(ZIP_OUTPUT_PATH, LEGACY_ZIP_PATH)
    shutil.copy2(ZIP_OUTPUT_PATH, ROOT_LEGACY_ZIP_PATH)
    print(f"Copied ZIP to root: {ROOT_ZIP_OUTPUT_PATH}")

    # Verify ZIP integrity
    print("\n--- Testing ZIP File Integrity ---")
    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "r") as test_zip:
        bad_file = test_zip.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupted file found in ZIP: {bad_file}")
        
        namelist = test_zip.namelist()
        print(f"Total files in archive: {len(namelist)}")
        print(f"Archive Size: {ZIP_OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")
        
        required_signatures = [
            "app.py", "train_model.py", "evaluate_model.py", "data_preprocessing.py",
            "project_config.py", "gradcam.py", "eda.py", "requirements.txt", "README.md",
            "dataset.csv", "cleaned_dataset.csv", "data_dictionary.csv",
            "eda_analysis.ipynb",
            "Project_Synopsis_Monika_Yadav_2400320109014.docx",
            "Crop_Disease_Detection_Synopsis_monika_2400320109014.docx",
            "Project_Synopsis_Monika_Yadav_2400320109014.pdf",
            "Project_Report_Monika_Yadav_2400320109014.docx",
            "Project_Report_Monika_Yadav_2400320109014.pdf",
            "models/class_labels.json", "models/disease_info.json",
            "reports/figures/dfd_level_0.png", "reports/figures/system_architecture.png",
            "reports/figures/confusion_matrix.png", "reports/figures/training_validation_curves.png"
        ]

        missing = []
        for sig in required_signatures:
            target = f"crop_disease_detection/{sig}"
            if target not in namelist:
                missing.append(sig)

        if missing:
            print(f"WARNING: Missing signatures: {missing}")
        else:
            print(">>> ALL REQUIRED PROJECT ARTIFACTS VERIFIED PRESENT AND VALID IN ZIP! <<<")


if __name__ == "__main__":
    create_submission_zip()
