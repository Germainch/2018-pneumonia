"""
Generate labels.csv for the RSNA 2018 Pneumonia Detection Challenge dataset.

Sources
-------
- pneumonia-challenge-annotations-adjudicated-kaggle_2018.json
    Adjudicated consensus labels from 18 radiologists (used for the Kaggle challenge).
    Contains multi-class opacity probability labels + bounding boxes for opacity regions.
    Covers all 30,000 images.

- pneumonia-challenge-annotations-original_2018.json
    Original binary labels: "Lung Opacity" / "No Lung Opacity".
    Covers 25,684 of the 30,000 images (strict subset of adjudicated).
    Used as a secondary reference column.

- pneumonia-challenge-dataset-mappings_2018.json
    Maps each image's original NIH img_id to DICOM UIDs (Study/Series/SOP)
    and includes the original NIH CXR8 disease labels.

- DICOM files (mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/)
    Provide patient metadata: PatientSex, PatientAge, ViewPosition (PA/AP).

Label logic (adjudicated, majority vote per image)
---------------------------------------------------
  sick     (label=1): Lung Opacity at any confidence (High / Med / Low Prob)
  not_sick (label=0): Normal  OR  No Lung Opacity / Not Normal
  Ties               → sick  (conservative / clinically safer)
  Only admin labels  → excluded from output (skipped=84)

Challenge encoding (PDE339-Pneumonia Detection)
  0 = Unknown  |  1 = Pneumonia present  |  2 = Pneumonia absent

Output: labels.csv
Columns
-------
  img_id              – original NIH CXR8 PNG filename
  SOPInstanceUID      – DICOM SOP UID
  StudyInstanceUID    – DICOM Study UID
  SeriesInstanceUID   – DICOM Series UID
  dicom_path          – absolute path to .dcm file
  patient_id          – DICOM PatientID
  patient_sex         – M / F (from DICOM)
  patient_age         – integer years (from DICOM)
  view_position       – PA (posteroanterior) or AP (anteroposterior)
  label               – 1 (sick) or 0 (not_sick)  [adjudicated majority vote]
  label_name          – "sick" or "not_sick"
  label_original      – 1/0 from original binary JSON, or "" if not present
  nih_orig_labels     – comma-separated original NIH disease tags (from mappings)
  bboxes              – JSON list of {x,y,width,height} opacity bounding boxes;
                        empty list [] for not_sick images
"""

import json
import os
import csv
from collections import defaultdict

import pydicom

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(DATASET_DIR, "../data/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330")

ADJUDICATED_FILE = os.path.join(DATASET_DIR, "../data/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json")
ORIGINAL_FILE    = os.path.join(DATASET_DIR, "../data/pneumonia-challenge-annotations-original_2018.json")
MAPPINGS_FILE    = os.path.join(DATASET_DIR, "../data/pneumonia-challenge-dataset-mappings_2018.json")
OUTPUT_FILE      = os.path.join(DATASET_DIR, "../data/labels.csv")

# ---------------------------------------------------------------------------
# Label sets  (adjudicated)
# ---------------------------------------------------------------------------
SICK_LABELS = {
    "Lung Opacity",
    "Lung Opacity (High Prob)",
    "Lung Opacity (Med Prob)",
    "Lung Opacity (Med Prob) ",
    "Lung Opacity (Low Prob)",
    "Lung Opacity (Low Prob) ",
}
NOT_SICK_LABELS = {"Normal", "No Lung Opacity / Not Normal"}
IGNORE_LABELS   = {
    "Question", "Question Addressed",
    "Exclude", "Adjudicate", "QA",
    "Flag", "Flag 2", "Flag 3", "Flag 4", "Flag 5",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_label_map(data):
    return {
        label["id"]: label["name"]
        for group in data["labelGroups"]
        for label in group["labels"]
    }


def dicom_path_for(mapping):
    return os.path.join(
        IMAGES_DIR,
        mapping["StudyInstanceUID"],
        mapping["SeriesInstanceUID"],
        mapping["SOPInstanceUID"] + ".dcm",
    )


def classify_adjudicated(sop, by_sop, label_map):
    """
    Returns (label, bboxes) where label is 1/0/-1 and bboxes is a list of dicts.
    -1 means only admin/QA annotations → skip.
    """
    entries = by_sop.get(sop, [])
    relevant = [(label_map.get(e["labelId"], e["labelId"]).strip(), e) for e in entries]
    relevant = [(name, e) for name, e in relevant if name not in IGNORE_LABELS]

    sick_count     = sum(1 for name, _ in relevant if name in SICK_LABELS)
    not_sick_count = sum(1 for name, _ in relevant if name in NOT_SICK_LABELS)

    if sick_count == 0 and not_sick_count == 0:
        return -1, []

    label = 1 if sick_count >= not_sick_count else 0

    bboxes = []
    if label == 1:
        for name, e in relevant:
            if name in SICK_LABELS and e.get("data"):
                d = e["data"]
                bboxes.append({
                    "x":      round(d["x"], 2),
                    "y":      round(d["y"], 2),
                    "width":  round(d["width"], 2),
                    "height": round(d["height"], 2),
                })

    return label, bboxes


def read_dicom_meta(path):
    """Extract patient metadata from a DICOM file. Returns dict."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        age = getattr(ds, "PatientAge", None)
        if age is not None:
            try:
                age = int(str(age).replace("Y", "").replace("y", "").strip())
            except (ValueError, TypeError):
                age = ""
        return {
            "patient_id":    getattr(ds, "PatientID",    ""),
            "patient_sex":   getattr(ds, "PatientSex",   ""),
            "patient_age":   age if age is not None else "",
            "view_position": getattr(ds, "ViewPosition", ""),
        }
    except Exception:
        return {"patient_id": "", "patient_sex": "", "patient_age": "", "view_position": ""}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading JSON files...")

    with open(ADJUDICATED_FILE) as f:
        adj_data = json.load(f)
    with open(ORIGINAL_FILE) as f:
        orig_data = json.load(f)
    with open(MAPPINGS_FILE) as f:
        mappings = json.load(f)

    # --- Adjudicated: group annotations by SOPInstanceUID ---
    adj_label_map = build_label_map(adj_data)
    adj_by_sop    = defaultdict(list)
    for ann in adj_data["datasets"][0]["annotations"]:
        sop = ann.get("SOPInstanceUID")
        if sop:
            adj_by_sop[sop].append(ann)

    # --- Original: simple binary label per SOPInstanceUID ---
    orig_label_map = build_label_map(orig_data)  # "Lung Opacity" / "No Lung Opacity"
    orig_by_sop    = {}
    for ann in orig_data["datasets"][0]["annotations"]:
        sop = ann.get("SOPInstanceUID")
        if sop:
            name = orig_label_map.get(ann["labelId"], "").strip()
            # Last annotation wins (they should be consistent)
            orig_by_sop[sop] = 1 if name == "Lung Opacity" else 0

    # --- Process each image ---
    print(f"Processing {len(mappings)} images and extracting DICOM metadata...")

    rows    = []
    skipped = 0

    for i, m in enumerate(mappings, 1):
        if i % 2000 == 0:
            print(f"  {i}/{len(mappings)}")

        sop = m["SOPInstanceUID"]
        label, bboxes = classify_adjudicated(sop, adj_by_sop, adj_label_map)

        if label == -1:
            skipped += 1
            continue

        path = dicom_path_for(m)
        meta = read_dicom_meta(path)

        rows.append({
            "img_id":           m["img_id"],
            "SOPInstanceUID":   sop,
            "StudyInstanceUID": m["StudyInstanceUID"],
            "SeriesInstanceUID":m["SeriesInstanceUID"],
            "dicom_path":       path,
            "patient_id":       meta["patient_id"],
            "patient_sex":      meta["patient_sex"],
            "patient_age":      meta["patient_age"],
            "view_position":    meta["view_position"],
            "label":            label,
            "label_name":       "sick" if label == 1 else "not_sick",
            "label_original":   orig_by_sop.get(sop, ""),
            "nih_orig_labels":  ", ".join(m.get("orig_labels", [])),
            "bboxes":           json.dumps(bboxes),
        })

    # --- Write CSV ---
    fieldnames = [
        "img_id", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID",
        "dicom_path", "patient_id", "patient_sex", "patient_age", "view_position",
        "label", "label_name", "label_original", "nih_orig_labels", "bboxes",
    ]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sick     = sum(1 for r in rows if r["label"] == 1)
    not_sick = sum(1 for r in rows if r["label"] == 0)
    with_orig = sum(1 for r in rows if r["label_original"] != "")

    print(f"\nDone.")
    print(f"  Total labeled : {len(rows)}")
    print(f"  Sick (1)      : {sick}")
    print(f"  Not-sick (0)  : {not_sick}")
    print(f"  Skipped       : {skipped}  (only admin/QA annotations)")
    print(f"  With original : {with_orig}  (have label from original JSON too)")
    print(f"  Output        : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
