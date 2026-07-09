import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import json
import zipfile
from pathlib import Path

from pattern_training import (
    segment_garment_only,
    clean_mask,
    find_outline,
    get_bounding_measurements,
    extract_clean_side_profile,
    extract_clean_neckline_profile,
    extract_clean_strap_profile,
    px_to_mm,
    draft_pattern,
    save_labels,
    prepare_printing,
)

DEFAULT_DPI = 250.0


def create_zip(file_paths, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in file_paths:
            archive.write(path, arcname=os.path.basename(path))
    return zip_path


def generate_pattern_outputs(image_path, neckline_style):
    temp_dir = tempfile.mkdtemp()
    exported_path = os.path.join(temp_dir, "outline.png")
    pattern_path = os.path.join(temp_dir, "pattern.png")
    mask_path = os.path.join(temp_dir, "mask_clean.png")
    label_path = os.path.join(temp_dir, "pattern.json")
    page_folder = os.path.join(temp_dir, "pattern_pages")
    os.makedirs(page_folder, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read uploaded image.")

    segmented_mask = segment_garment_only(image_path, exported_path)

    try:
        binary = clean_mask(segmented_mask, flatten_top=False)
        outline = find_outline(binary, image, exported_path)
        measurements, _ = get_bounding_measurements(outline)

        chest_y = measurements["image_y"].get("chest_y_px")
        shoulder_y = measurements["image_y"].get("shoulder_y_px")

        right_neckline_profile = extract_clean_neckline_profile(
            outline,
            chest_y=chest_y,
            shoulder_y=shoulder_y,
            side="right",
        )

        right_strap_profile = extract_clean_strap_profile(
            outline,
            side="right",
        )

        st.info("Natural neckline and strap profile detected.")

    except ValueError as exc:
        st.warning(f"Natural top detection failed: {exc}")
        st.info("Using simplified top cleanup instead.")

        binary = clean_mask(segmented_mask, flatten_top=True)
        outline = find_outline(binary, image, exported_path)
        measurements, _ = get_bounding_measurements(outline)
        right_neckline_profile = None
        right_strap_profile = None

    cv2.imwrite(mask_path, binary)

    right_profile = extract_clean_side_profile(
        outline,
        side="right",
    )

    measurements["right_profile"] = right_profile
    measurements["right_neckline_profile"] = right_neckline_profile
    measurements["right_strap_profile"] = right_strap_profile

    measurements_mm = px_to_mm(measurements)
    measurements_mm["right_profile"] = right_profile
    measurements_mm["right_neckline_profile"] = right_neckline_profile
    measurements_mm["right_strap_profile"] = right_strap_profile

    canvas, _ = draft_pattern(
        measurements_mm,
        pattern_path,
        neckline_style=neckline_style,
        dpi=DEFAULT_DPI,
    )

    save_labels(measurements_mm, label_path)

    page_files = prepare_printing(
        Path(image_path).stem,
        canvas,
        page_folder,
        dpi=DEFAULT_DPI,
    )

    zip_path = os.path.join(temp_dir, f"{Path(image_path).stem}_pattern_pages.zip")
    create_zip(page_files, zip_path)

    return {
        "exported_path": exported_path,
        "pattern_path": pattern_path,
        "mask_path": mask_path,
        "label_path": label_path,
        "page_files": page_files,
        "zip_path": zip_path,
        "measurements_mm": measurements_mm,
    }


# -------------------------------------------------
# Streamlit page setup
# -------------------------------------------------

st.set_page_config(
    page_title="Hailey Pattern Generator",
    layout="wide",
)

st.title("Hailey Custom Garment Pattern Generator")

st.write(
    "Upload a garment image. The app will segment the garment, "
    "extract its shape and measurements, and generate a sewing pattern."
)

uploaded_file = st.file_uploader(
    "Upload garment image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

neckline_style = st.radio(
    "Select the neckline curve style:",
    options=["Auto", "Straight (linear)", "Soft scoop (exponential)"],
    index=0,
    help=(
        "Auto uses the detected top shape; "
        "Straight uses a simple line; "
        "Soft scoop uses a gentle curved neckline."
    ),
)

if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    suffix = Path(uploaded_file.name).suffix
    image_path = os.path.join(temp_dir, "input" + suffix)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    uploaded_image = cv2.imread(image_path)
    if uploaded_image is None:
        st.error("Could not read uploaded image.")
    else:
        uploaded_rgb = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
        st.subheader("Uploaded Image")
        st.image(uploaded_rgb, use_container_width=True)

        if st.button("Generate Pattern"):
            with st.spinner("Analyzing garment and generating pattern..."):
                try:
                    results = generate_pattern_outputs(image_path, neckline_style)

                    st.success("Pattern successfully generated.")

                    st.subheader("Processing Results")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("### Cleaned Mask")
                        mask_img = cv2.imread(results["mask_path"], cv2.IMREAD_GRAYSCALE)
                        st.image(mask_img, use_container_width=True)

                    with col2:
                        st.markdown("### Detected Outline")
                        outline_img = cv2.imread(results["exported_path"])
                        if outline_img is not None:
                            outline_rgb = cv2.cvtColor(outline_img, cv2.COLOR_BGR2RGB)
                            st.image(outline_rgb, use_container_width=True)

                    with col3:
                        st.markdown("### Generated Pattern")
                        pattern_img = cv2.imread(results["pattern_path"])
                        if pattern_img is not None:
                            pattern_rgb = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2RGB)
                            st.image(pattern_rgb, use_container_width=True)

                    st.subheader("Detected Measurements")
                    points = results["measurements_mm"].get("points", {})
                    measurement_names = {
                        "body_length_mm": "Body Length",
                        "side_length_mm": "Side Length",
                        "hem_width_mm": "Hem Width",
                        "shoulder_width_mm": "Shoulder Width",
                        "shoulder_length_mm": "Shoulder Length",
                        "chest_width_mm": "Chest Width",
                        "chest_length_mm": "Chest Height",
                        "waist_width_mm": "Waist Width",
                        "waist_length_mm": "Waist Height",
                        "strap_width1_mm": "Strap Width",
                        "strap_length_mm": "Strap Length",
                    }
                    for key, display_name in measurement_names.items():
                        value = points.get(key)
                        if value is not None:
                            st.write(f"**{display_name}:** {value:.1f} mm")

                    st.subheader("Page Output")
                    st.write(f"Generated {len(results['page_files'])} page(s) at {int(DEFAULT_DPI)} DPI.")
                    st.write("Page labels are added in left-to-right, top-to-bottom order.")

                    page_cols = st.columns(3)
                    for idx, page_file in enumerate(results["page_files"]):
                        with page_cols[idx % 3]:
                            page_img = cv2.imread(page_file)
                            if page_img is not None:
                                page_rgb = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)
                                st.image(page_rgb, caption=Path(page_file).name, use_column_width=True)

                    st.subheader("Downloads")
                    col_download1, col_download2, col_download3 = st.columns(3)

                    with col_download1:
                        with open(results["pattern_path"], "rb") as f:
                            st.download_button(
                                label="Download Pattern Image",
                                data=f.read(),
                                file_name="generated_pattern.png",
                                mime="image/png",
                            )

                    with col_download2:
                        with open(results["label_path"], "rb") as f:
                            st.download_button(
                                label="Download Pattern Measurements",
                                data=f.read(),
                                file_name="pattern_measurements.json",
                                mime="application/json",
                            )

                    with col_download3:
                        with open(results["zip_path"], "rb") as f:
                            st.download_button(
                                label="Download Pattern Pages ZIP",
                                data=f.read(),
                                file_name=f"{Path(uploaded_file.name).stem}_pattern_pages.zip",
                                mime="application/zip",
                            )

                except Exception as e:
                    st.error(f"Pattern generation failed: {e}")
                    st.exception(e)
