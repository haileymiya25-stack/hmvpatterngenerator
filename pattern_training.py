#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:40:15 2026

@author: haileyvan
"""

'''binarize image and remove noise, print out basic shape'''

import numpy as np
import cv2
import os
import json
import sys
from transformers import pipeline
from pathlib import Path

DEFAULT_DPI = 250.0
DEFAULT_HEM_WIDTH_MM = 380.0


segmenter = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes")

def segment_garment_only(image_path, output_path):
    results = segmenter(image_path)
    
    # Labels this model recognizes: Upper-clothes, Pants, Skirt, Dress etc.
    garment_labels = ["Upper-clothes", "Dress", "Skirt"]
    
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    for r in results:
        if r["label"] in garment_labels:
            mask = np.array(r["mask"])
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    cv2.imwrite(output_path.replace(".png", "_mask.png"), combined_mask)
    return combined_mask

def binarize(image, threshold=250):
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.GaussianBlur(bw, (7, 7), 0)
    
    # Keep brightest pixels (white background)
    _, binary = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)
    
    # Invert so garment = white, background = black
    binary = cv2.bitwise_not(binary)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    return binary

def keep_largest_component(binary):
    """Drop everything except the largest connected blob (the garment)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Label 0 is background — find largest non-background component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(binary)
    mask[labels == largest_label] = 255
    return mask

def clean_mask(mask, flatten_top=True):
    h, w = mask.shape
    
    # 1. Fill internal holes first
    filled = mask.copy()
    flood = np.zeros((h+2, w+2), dtype=np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    mask = cv2.bitwise_or(mask, filled_inv)

    # 2. Keep only the largest blob before any morphology
    mask = keep_largest_component(mask)

    # 3. Gentle close to fill small gaps — NOT large enough to merge arms
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    # 4. Open to remove noise bumps on edges
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 5. Keep largest component again after morphology
    mask = keep_largest_component(mask)

    if flatten_top:
        # 6. Crop top 20% — neckline/strap area is noisy,
        #    replace with straight horizontal cut.
        top_cutoff = int(h * 0.20)

        # Find the leftmost and rightmost white pixel at the cutoff line
        row = mask[top_cutoff, :]
        white_cols = np.where(row > 128)[0]
        if len(white_cols) >= 2:
            left_x  = white_cols[0]
            right_x = white_cols[-1]
            # Fill everything above cutoff with black
            mask[:top_cutoff, :] = 0
            # Draw a clean straight top edge
            mask[top_cutoff-2:top_cutoff+2, left_x:right_x] = 255

    return mask
 
def find_outline(binary, image, output_path, epsilon_frac=0.002, min_area_frac=0.1):
    clean_mask = keep_largest_component(binary)
    contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Filter out small noise contours by area
    image_area = image.shape[0] * image.shape[1]
    min_area = min_area_frac * image_area
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(contour, True)
    smoothed = cv2.approxPolyDP(contour, epsilon, True)
    h,w = image.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(canvas, [smoothed], 0, (0, 0, 255), 2)
    cv2.imwrite(output_path, canvas)
    
    return contour  

def extract_clean_side_profile(contour, side="right", num_samples=80):
    """
    Extract a smoothed side profile from a contour.

    Returns normalized points:
    x_norm = side variation
    y_norm = 0 at hem, 1 at top
    """

    import numpy as np

    pts = np.asarray(contour)

    # OpenCV contours are usually (N, 1, 2)
    if pts.ndim == 3:
        pts = pts[:, 0, :]

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected contour shape (N, 2), got {pts.shape}")

    if len(pts) < 5:
        raise ValueError(f"Contour has too few points: {len(pts)}")

    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()
    height = y_max - y_min

    if height == 0:
        raise ValueError("Contour height is zero.")

    sampled = []

    y_samples = np.linspace(y_min, y_max, num_samples)

    for y in y_samples:
        nearby = None

        # Try larger and larger tolerances until points are found
        for tolerance in [3, 5, 8, 12, 20]:
            nearby_try = pts[np.abs(pts[:, 1] - y) <= tolerance]

            if len(nearby_try) >= 2:
                nearby = nearby_try
                break

        if nearby is None:
            continue

        if side == "right":
            x = nearby[:, 0].max()
        else:
            x = nearby[:, 0].min()

        sampled.append([x, y])

    sampled = np.array(sampled, dtype=float)

    print("side profile sampled points:", len(sampled))

    if len(sampled) < 5:
        raise ValueError(
            "Not enough side points were found. "
            "Use the original contour, not approxPolyDP smoothed contour."
        )

    # Sort top to bottom
    sampled = sampled[np.argsort(sampled[:, 1])]

    # Smooth x-values
    x_vals = sampled[:, 0].copy()

    window = 7
    if len(x_vals) >= window:
        kernel = np.ones(window) / window
        x_smooth = np.convolve(x_vals, kernel, mode="same")

        half = window // 2
        x_smooth[:half] = x_vals[:half]
        x_smooth[-half:] = x_vals[-half:]
    else:
        x_smooth = x_vals

    sampled[:, 0] = x_smooth

    # Normalize
    x_min = sampled[:, 0].min()
    x_max = sampled[:, 0].max()
    x_range = x_max - x_min

    if x_range == 0:
        x_range = 1

    normalized = []

    for x, y in sampled:
        x_norm = (x - x_min) / x_range
        y_norm = 1 - ((y - y_min) / height)
        normalized.append([float(x_norm), float(y_norm)])

    normalized = np.array(normalized)

    def point_at_y(target_y):
        idx = np.argmin(np.abs(normalized[:, 1] - target_y))
        return normalized[idx].tolist()

    hem = point_at_y(0.00)
    chest = point_at_y(0.75)
    top = point_at_y(1.00)

    # Waist = most inward point between 35% and 65% up the body
    waist_region = normalized[
        (normalized[:, 1] >= 0.35) &
        (normalized[:, 1] <= 0.65)
    ]

    if len(waist_region) > 0:
        if side == "right":
            waist = waist_region[np.argmin(waist_region[:, 0])].tolist()
        else:
            waist = waist_region[np.argmax(waist_region[:, 0])].tolist()
    else:
        waist = point_at_y(0.50)

    return {
        "hem": hem,
        "waist": waist,
        "chest": chest,
        "top": top,
        "curve_points": normalized.tolist()
    }


def extract_clean_neckline_profile(contour, chest_y, shoulder_y, side="right", num_samples=60):
    """
    Extract a smoothed inner neckline profile between chest and shoulder.

    Returns normalized points:
    x_norm = neckline side variation
    y_norm = 0 at chest, 1 at shoulder
    """

    pts = np.asarray(contour)

    if pts.ndim == 3:
        pts = pts[:, 0, :]

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected contour shape (N, 2), got {pts.shape}")

    if chest_y is None or shoulder_y is None:
        raise ValueError("Need chest_y and shoulder_y to extract neckline profile.")

    y_low = min(chest_y, shoulder_y)
    y_high = max(chest_y, shoulder_y)
    height = y_high - y_low

    if height == 0:
        raise ValueError("Neckline height is zero.")

    sampled = []
    y_samples = np.linspace(y_low, y_high, num_samples)

    for y in y_samples:
        nearby = None

        for tolerance in [3, 5, 8, 12, 20]:
            nearby_try = pts[np.abs(pts[:, 1] - y) <= tolerance]

            if len(nearby_try) >= 4:
                nearby = nearby_try
                break

        if nearby is None:
            continue

        xs = np.sort(nearby[:, 0])
        gaps = np.diff(xs)

        if len(gaps) == 0:
            continue

        split = np.argmax(gaps)
        left_cluster = xs[:split + 1]
        right_cluster = xs[split + 1:]

        if len(left_cluster) == 0 or len(right_cluster) == 0:
            continue

        if side == "right":
            x = right_cluster.min()
        else:
            x = left_cluster.max()

        sampled.append([x, y])

    sampled = np.array(sampled, dtype=float)

    print("neckline profile sampled points:", len(sampled))

    if len(sampled) < 5:
        raise ValueError(
            "Not enough neckline points were found. "
            "The top cleanup may be flattening the neckline before extraction."
        )

    sampled = sampled[np.argsort(sampled[:, 1])]

    x_vals = sampled[:, 0].copy()
    window = 7

    if len(x_vals) >= window:
        kernel = np.ones(window) / window
        x_smooth = np.convolve(x_vals, kernel, mode="same")

        half = window // 2
        x_smooth[:half] = x_vals[:half]
        x_smooth[-half:] = x_vals[-half:]
    else:
        x_smooth = x_vals

    sampled[:, 0] = x_smooth

    x_min = sampled[:, 0].min()
    x_max = sampled[:, 0].max()
    x_range = x_max - x_min

    if x_range == 0:
        x_range = 1

    normalized = []

    for x, y in sampled:
        x_norm = (x - x_min) / x_range
        y_norm = 1 - ((y - y_low) / height)
        normalized.append([float(x_norm), float(y_norm)])

    normalized = np.array(normalized)

    def point_at_y(target_y):
        idx = np.argmin(np.abs(normalized[:, 1] - target_y))
        return normalized[idx].tolist()

    return {
        "chest": point_at_y(0.00),
        "shoulder": point_at_y(1.00),
        "curve_points": normalized.tolist()
    }


def extract_clean_strap_profile(contour, side="right", num_samples=60):
    """
    Extract a simplified strap profile from the upper contour.

    Returns normalized inner/outer strap edges, plus width and lean signals.
    y_norm = 0 at strap bottom, 1 at strap top.
    """

    pts = np.asarray(contour)

    if pts.ndim == 3:
        pts = pts[:, 0, :]

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected contour shape (N, 2), got {pts.shape}")

    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()
    total_height = y_max - y_min

    if total_height == 0:
        raise ValueError("Contour height is zero.")

    y_scan_bottom = y_min + total_height * 0.35
    y_samples = np.linspace(y_min, y_scan_bottom, num_samples)
    sampled = []

    for y in y_samples:
        nearby = None

        for tolerance in [3, 5, 8, 12, 20]:
            nearby_try = pts[np.abs(pts[:, 1] - y) <= tolerance]

            if len(nearby_try) >= 4:
                nearby = nearby_try
                break

        if nearby is None:
            continue

        xs = np.sort(nearby[:, 0])
        gaps = np.diff(xs)

        if len(gaps) == 0:
            continue

        split = np.argmax(gaps)
        left_cluster = xs[:split + 1]
        right_cluster = xs[split + 1:]

        if len(left_cluster) < 2 or len(right_cluster) < 2:
            continue

        cluster = right_cluster if side == "right" else left_cluster
        inner_x = cluster.min() if side == "right" else cluster.max()
        outer_x = cluster.max() if side == "right" else cluster.min()
        width = abs(outer_x - inner_x)

        if width < 5:
            continue

        sampled.append([inner_x, outer_x, y, width])

    sampled = np.array(sampled, dtype=float)

    print("strap profile sampled points:", len(sampled))

    if len(sampled) < 5:
        raise ValueError("Not enough strap points were found.")

    widths = sampled[:, 3]
    median_width = np.median(widths)
    keep = np.abs(widths - median_width) <= max(8, median_width * 0.60)
    sampled = sampled[keep]

    if len(sampled) < 5:
        raise ValueError("Not enough stable strap points were found.")

    sampled = sampled[np.argsort(sampled[:, 2])]

    for col in [0, 1]:
        vals = sampled[:, col].copy()
        window = 7

        if len(vals) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(vals, kernel, mode="same")

            half = window // 2
            smooth[:half] = vals[:half]
            smooth[-half:] = vals[-half:]
            sampled[:, col] = smooth

    y_top = sampled[:, 2].min()
    y_bottom = sampled[:, 2].max()
    height = y_bottom - y_top

    if height == 0:
        raise ValueError("Strap height is zero.")

    if height < total_height * 0.06 or height < 20:
        raise ValueError("Detected strap profile is too short.")

    top_rows = sampled[sampled[:, 2] <= y_top + height * 0.25]
    bottom_rows = sampled[sampled[:, 2] >= y_bottom - height * 0.25]

    top_inner = float(np.median(top_rows[:, 0]))
    top_outer = float(np.median(top_rows[:, 1]))
    bottom_inner = float(np.median(bottom_rows[:, 0]))
    bottom_outer = float(np.median(bottom_rows[:, 1]))

    top_center = (top_inner + top_outer) / 2
    bottom_center = (bottom_inner + bottom_outer) / 2
    pixel_width = float(np.median(np.abs(sampled[:, 1] - sampled[:, 0])))

    if pixel_width < 8:
        raise ValueError("Detected strap profile is too narrow.")

    lean_norm = float((top_center - bottom_center) / max(pixel_width, 1))
    top_width_ratio = float(abs(top_outer - top_inner) / max(pixel_width, 1))

    edge_points = []

    for inner_x, outer_x, y, width in sampled:
        y_norm = 1 - ((y - y_top) / height)
        edge_points.append({
            "inner_x_norm": float((inner_x - bottom_inner) / max(pixel_width, 1)),
            "outer_x_norm": float((outer_x - bottom_outer) / max(pixel_width, 1)),
            "y_norm": float(y_norm)
        })

    return {
        "height_px": float(height),
        "width_px": pixel_width,
        "lean_norm": lean_norm,
        "top_width_ratio": top_width_ratio,
        "edge_points": edge_points
    }


def fit_curve_segment(pts, y_start, y_end, degree=2, num_points=100):
    """
    Fits a polynomial curve to contour points between two y positions.
    Handles y_start > y_end safely.
    """

    if y_start is None or y_end is None:
        print("Cannot fit curve: y_start or y_end is None")
        return None

    # Make sure lower_y <= upper_y
    y_low = min(y_start, y_end)
    y_high = max(y_start, y_end)

    # Get contour points in that vertical region
    mask = (pts[:, 1] >= y_low) & (pts[:, 1] <= y_high)
    segment = pts[mask]

    if len(segment) < degree + 1:
        print(f"Not enough points between y={y_low:.0f} and y={y_high:.0f}")
        return None

    x = segment[:, 0].astype(float)
    y = segment[:, 1].astype(float)

    coeffs = np.polyfit(y, x, degree)

    y_smooth = np.linspace(y_low, y_high, num_points)
    x_smooth = np.polyval(coeffs, y_smooth)

    curve_pts = np.column_stack([x_smooth, y_smooth])

    diffs = np.diff(curve_pts, axis=0)
    arc_length = np.sum(np.sqrt((diffs ** 2).sum(axis=1)))

    return {
        "coeffs": coeffs,
        "curve_pts": curve_pts,
        "arc_length": round(arc_length, 1)
    }

def get_bounding_measurements(contour):
    pts = contour.squeeze()
    x, y, w, h = cv2.boundingRect(contour)
    top_y = pts[:, 1].min()
    bottom_y = pts[:, 1].max()
    total_height = bottom_y - top_y
    shoulder_y = top_y + 0.25*total_height


    def width_at_y(y_target, tolerance=15):
        nearby = pts[np.abs(pts[:, 1]- y_target) < tolerance]
        if len(nearby) <2: 
            return None, None, None
        else: 
            left_x = nearby[:, 0].min()
            right_x = nearby[:, 0].max()
            return left_x, right_x, right_x - left_x
    
    
    def strap_widths_at_y(y_target, tolerance=25): 
        nearby = pts[np.abs(pts[:, 1] - y_target) < tolerance]
        if len(nearby) < 4:
            return None, None
        x_coo = np.sort(nearby[:,0]) # all x_values at a given y point
        gaps = np.diff(x_coo)
        split = np.argmax(gaps)
        left = x_coo[:split +1]
        right = x_coo[split+1:]
        strap1 = left.max() - left.min()
        strap2 = right.max() - right.min()
        return strap1, strap2
    
    
    list1 = []
    list2 = []
    for i in range(int(top_y), int(shoulder_y), 1):
        width1, width2 = strap_widths_at_y(i)
        list1.append(width1)
        list2.append(width2)
    
    list1 = [w for w in list1 if w is not None and w != 0]
    list2 = [w for w in list2 if w is not None and w != 0]
    
    final_strap_width = None
    for w1 in list1:
        for w2 in list2:
            if w1 is None or w2 is None:
                continue
            if w1 > 10 and w2 > 10: 
                if abs(w1 - w2) <= 3:
                    print(f"matched: w1={w1}, w2={w2}")
                    final_strap_width = round((w1+w2)/2, 1)
                    break
        if final_strap_width is not None:
            break
        
    '''define widths'''
        
    widths = []
    starts = []
    lengths = []
    fractions = []
    for step in range(0, 100, 1): 
        y_height = top_y + total_height*(step*0.01)
        start, end, width = width_at_y(y_height)
        #waist must be at least 100 pixels
        if width is not None and width > 100:
            widths.append(width)
            starts.append(start)
            lengths.append(y_height)
            fractions.append((y_height - top_y) / total_height)

    widths = np.array(widths)
    starts = np.array(starts)
    lengths = np.array(lengths)
    fractions = np.array(fractions)

    def nearest_index(target_frac):
        return int(np.argmin(np.abs(fractions - target_frac)))

    def min_width_index(frac_min, frac_max):
        region = np.where((fractions >= frac_min) & (fractions <= frac_max))[0]

        if len(region) == 0:
            return nearest_index((frac_min + frac_max) / 2)

        return int(region[np.argmin(widths[region])])

    def max_width_index(frac_min, frac_max):
        region = np.where((fractions >= frac_min) & (fractions <= frac_max))[0]

        if len(region) == 0:
            return nearest_index((frac_min + frac_max) / 2)

        return int(region[np.argmax(widths[region])])

    shoulder_location = max_width_index(0.10, 0.35)
    chest_location = nearest_index(0.35)
    waist_location = min_width_index(0.45, 0.75)
    hem_location = nearest_index(0.98)

    shoulder_width = float(widths[shoulder_location])
    chest_width = float(widths[chest_location])
    waist_width = float(widths[waist_location])
    hem_width = float(widths[hem_location])

    shoulder_y_px = float(lengths[shoulder_location])
    chest_y_px = float(lengths[chest_location])
    waist_y_px = float(lengths[waist_location])
    hem_y_px = float(lengths[hem_location])

    shoulder_length = bottom_y - shoulder_y_px
    chest_length = bottom_y - chest_y_px
    waist_length = bottom_y - waist_y_px
    hem_length = 0
     
    
    '''Calculating strap length'''
    strapies = []
    strap_length = None

    if final_strap_width is not None:
        scan_bottom = top_y + total_height * 0.35
        tol = max(12, final_strap_width * 0.75)

        for check in np.arange(top_y, scan_bottom, 1):
            w1, w2 = strap_widths_at_y(check)

            if w1 is None or w2 is None:
                continue

            left_matches = (final_strap_width - tol) < w1 < (final_strap_width + tol)
            right_matches = (final_strap_width - tol) < w2 < (final_strap_width + tol)

            if left_matches and right_matches:
                strapies.append(check)

        if len(strapies) >= 2:
            strap_length = max(strapies) - min(strapies)

    print("strapies=", strapies)
    
    center_points = []
    side_points = []
    for step in range(0, 100, 1):
        y_height = top_y + total_height * (step * 0.01)
        left_x, right_x, width = width_at_y(y_height)
        if left_x is not None and right_x is not None:
            mid_x = (left_x + right_x) / 2
            center_points.append((mid_x, y_height))
            side_points.append((left_x, y_height))
    
    center_points = np.array(center_points)
    side_points = np.array(side_points)
    
    if len(center_points) >= 2:
        diffs = np.diff(center_points, axis=0)
        body_length = np.sum(np.sqrt((diffs ** 2).sum(axis=1)))
    else:
        body_length = None
    
    if len(side_points) >=2: 
        diffs = np.diff(side_points, axis=0)
        side_length = np.sum(np.sqrt((diffs ** 2).sum(axis=1)))
    else:
        side_length = None
    
    #detecting_curve
    sc = fit_curve_segment(pts, shoulder_y_px, chest_y_px, degree=2)
    cw = fit_curve_segment(pts, chest_y_px, waist_y_px, degree=2)
    wh = fit_curve_segment(pts, waist_y_px, hem_y_px, degree=2)
    
    measurements = {
        "points": {
            "body_length_px":     round(body_length, 1)      if body_length is not None else None,
            "shoulder_width_px":  round(shoulder_width, 1)   if shoulder_width   else None,
            "shoulder_length_px":  round(shoulder_length, 1)   if shoulder_length  else None,
            "chest_width_px":     round(chest_width, 1)      if chest_width      else None,
            "chest_length_px":     round(chest_length, 1)      if chest_length   else None,
            "waist_width_px":     round(waist_width, 1)      if waist_width      else None,
            "waist_length_px":     round(waist_length, 1)      if waist_length      else None,
            "hem_width_px":       round(hem_width, 1)        if hem_width        else None,
            "hem_length_px":       round(hem_length, 1)        if hem_length is not None else None,
            "side_length_px":     round(side_length, 1)      if side_length      else None,
            "strap_width1_px":    round(final_strap_width, 1) if final_strap_width else None,
            "strap_width2_px":    round(final_strap_width, 1) if final_strap_width else None,
            "strap_length_px":    round(strap_length, 1)     if strap_length     else None,
        },
        "shoulder_chest_curve": {
            "coeffs":     sc["coeffs"].tolist() if sc else None,
            "curve_pts":  sc["curve_pts"].tolist() if sc else None,
            "arc_length": sc["arc_length"]         if sc else None,
        },
        "chest_waist_curve": {
            "coeffs":     cw["coeffs"].tolist() if cw else None,
            "curve_pts":  cw["curve_pts"].tolist() if cw else None,
            "arc_length": cw["arc_length"]         if cw else None,
        },
        "waist_hem_curve": {
            "coeffs":     wh["coeffs"].tolist() if wh else None,
            "curve_pts":  wh["curve_pts"].tolist() if wh else None,
            "arc_length": wh["arc_length"]         if wh else None,
        },
        "image_y": {
            "shoulder_y_px": shoulder_y_px,
            "chest_y_px": chest_y_px,
            "waist_y_px": waist_y_px,
            "hem_y_px": hem_y_px,
        }
    }
    return measurements, pts

def px_to_mm(measurements_px, px_per_mm=None, reference_width_mm=DEFAULT_HEM_WIDTH_MM):
    hem_px = measurements_px["points"].get("hem_width_px")

    if px_per_mm is not None:
        scale = 1.0 / float(px_per_mm)   # mm per pixel
    elif hem_px:
        scale = reference_width_mm / hem_px   # mm per pixel
    else:
        raise ValueError("Need hem_width_px or px_per_mm.")

    if os.environ.get("PATTERN_DEBUG_SCALE"):
        print(
            f"[debug] px_to_mm scale={scale:.4f} mm/px, hem_width_px={hem_px}, "
            f"reference_width_mm={reference_width_mm}"
        )

    result = {"points": {}}

    # Convert point measurements
    for k, v in measurements_px["points"].items():
        k_new = k.replace("_px", "_mm")
        result["points"][k_new] = round(v * scale, 1) if v is not None else None

    # Convert curve sections safely
    curve_sections = [
        "shoulder_chest_curve",
        "chest_waist_curve",
        "waist_hem_curve"
    ]

    for sec_name in curve_sections:
        sec_data = measurements_px.get(sec_name)

        result[sec_name] = {
            "coeffs": None,
            "curve_pts_mm": None,
            "arc_length_mm": None
        }

        if sec_data is None:
            continue

        result[sec_name]["coeffs"] = sec_data.get("coeffs")

        curve_pts = sec_data.get("curve_pts")
        if curve_pts is not None:
            result[sec_name]["curve_pts_mm"] = [
                [round(pt[0] * scale, 1), round(pt[1] * scale, 1)]
                for pt in curve_pts
            ]

        arc = sec_data.get("arc_length")
        if arc is not None:
            result[sec_name]["arc_length_mm"] = round(arc * scale, 1)

    return result


def save_labels(measurements_mm, output_path):
    label = {
        "front": {
            "body_length":   measurements_mm["points"].get("body_length_mm"),
            "side_length":   measurements_mm["points"].get("side_length_mm"),
            "hem_width":     measurements_mm["points"].get("hem_width_mm"),
            "hem_length":     measurements_mm["points"].get("hem_length_mm"),
            "shoulder_width": measurements_mm["points"].get("shoulder_width_mm"),
            "shoulder_length": measurements_mm["points"].get("shoulder_length_mm"),
            "chest_width":   measurements_mm["points"].get("chest_width_mm"),
            "chest_length": measurements_mm["points"].get("chest_length_mm"),
            "waist_width":   measurements_mm["points"].get("waist_width_mm"),
            "waist_length":   measurements_mm["points"].get("waist_length_mm"),
        },
        "strap": {
            "width":  measurements_mm["points"].get("strap_width1_mm"),
            "length": measurements_mm["points"].get("strap_length_mm"),
        }, 
        "shoulder_chest_curve": {
            "coeffs":     measurements_mm["shoulder_chest_curve"].get("coeffs"),
            "curve_pts":  measurements_mm["shoulder_chest_curve"].get("curve_pts_mm"),
            "arc_length": measurements_mm["shoulder_chest_curve"].get("arc_length_mm"),
        },
        "chest_waist_curve": {
            "coeffs":     measurements_mm["chest_waist_curve"].get("coeffs"),
            "curve_pts":  measurements_mm["chest_waist_curve"].get("curve_pts_mm"),
            "arc_length": measurements_mm["chest_waist_curve"].get("arc_length_mm"),
        },
        "waist_hem_curve": {
            "coeffs":     measurements_mm["waist_hem_curve"].get("coeffs"),
            "curve_pts":  measurements_mm["waist_hem_curve"].get("curve_pts_mm"),
            "arc_length": measurements_mm["waist_hem_curve"].get("arc_length_mm"),
        },
        "right_profile": measurements_mm.get("right_profile"),
        "right_neckline_profile": measurements_mm.get("right_neckline_profile"),
        "right_strap_profile": measurements_mm.get("right_strap_profile")
        
    }
    with open(output_path, "w") as f:
        json.dump(label, f, indent=4)
    print(f"Saved: {output_path}")


def draft_pattern(measurements_mm, output_path, neckline_style="Auto", dpi=DEFAULT_DPI):
    import cv2
    import numpy as np

    print(f"[debug] draft_pattern dpi={dpi}")
    # -------------------------------------------------
    # Setup
    # -------------------------------------------------
    scale = dpi / 25.4  # pixels per mm

    def mm(v):
        return int(round((v or 0) * scale))

    page_width = int(round(8 * dpi))
    page_height = int(round(11 * dpi))

    p = measurements_mm
    front = p.get("front")
    strap = p.get("strap")

    if front is None:
        points = p.get("points", {})
        front = {
            "body_length": points.get("body_length_mm"),
            "side_length": points.get("side_length_mm"),
            "hem_width": points.get("hem_width_mm"),
            "hem_length": points.get("hem_length_mm"),
            "shoulder_width": points.get("shoulder_width_mm"),
            "shoulder_length": points.get("shoulder_length_mm"),
            "chest_width": points.get("chest_width_mm"),
            "chest_length": points.get("chest_length_mm"),
            "waist_width": points.get("waist_width_mm"),
            "waist_length": points.get("waist_length_mm"),
        }

    if strap is None:
        points = p.get("points", {})
        strap = {
            "width": points.get("strap_width1_mm"),
            "length": points.get("strap_length_mm"),
        }

    # -------------------------------------------------
    # Measurements from extracted image
    # -------------------------------------------------
    body_length = front.get("body_length") or 500
    side_length = front.get("side_length") or body_length * 0.85

    hem_width = front.get("hem_width") or 380
    waist_width = front.get("waist_width") or hem_width * 0.85
    chest_width = front.get("chest_width") or hem_width * 0.95
    shoulder_width = front.get("shoulder_width") or chest_width

    waist_length = front.get("waist_length") or side_length * 0.50
    chest_length = front.get("chest_length") or side_length * 0.75
    shoulder_length = front.get("shoulder_length") or side_length

    detected_strap_width = strap.get("width")
    detected_strap_length = strap.get("length")
    strap_width = detected_strap_width or 35

    sa = 5  # seam allowance in mm

    # Half pattern widths
    # x = 0 is center front / fold line
    hw = hem_width / 2
    ww = waist_width / 2
    cw = chest_width / 2
    sw = shoulder_width / 2

    # Prevent strap from becoming impossible
    strap_width = min(strap_width, sw * 0.75)

    default_strap_length = max(70, side_length * 0.22)
    min_strap_length = max(45, side_length * 0.10)
    max_strap_length = max(min_strap_length + 10, side_length * 0.42)

    if detected_strap_length is None:
        strap_length = default_strap_length
    else:
        strap_length = np.clip(detected_strap_length, min_strap_length, max_strap_length)

    # Canvas will be initialized after pattern landmarks are computed.
    canvas = None
    x0 = 0
    y0 = 0

    def line(p1, p2, color=(0, 0, 0), thickness=4):
        cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    def dot(p, r=8, color=(0, 0, 255)):
        cv2.circle(canvas, p, r, color, -1, lineType=cv2.LINE_AA)

    def label(p, text, offset=(12, -8), size=0.65):
        cv2.putText(
            canvas,
            text,
            (p[0] + offset[0], p[1] + offset[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (60, 60, 60),
            2
        )

    def quadratic_bezier(p1, ctrl, p2, color=(0, 0, 0), thickness=4, n=100):
        pts = []

        for t in np.linspace(0, 1, n):
            x = (
                (1 - t) ** 2 * p1[0]
                + 2 * (1 - t) * t * ctrl[0]
                + t ** 2 * p2[0]
            )
            y = (
                (1 - t) ** 2 * p1[1]
                + 2 * (1 - t) * t * ctrl[1]
                + t ** 2 * p2[1]
            )
            pts.append([int(x), int(y)])

        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)

    def quadratic_points(p1, ctrl, p2, n=100):
        pts = []

        for t in np.linspace(0, 1, n):
            x = (
                (1 - t) ** 2 * p1[0]
                + 2 * (1 - t) * t * ctrl[0]
                + t ** 2 * p2[0]
            )
            y = (
                (1 - t) ** 2 * p1[1]
                + 2 * (1 - t) * t * ctrl[1]
                + t ** 2 * p2[1]
            )
            pts.append((int(x), int(y)))

        return pts

    def draw_neckline_curve(p1, p2, style="exponential", n=100, color=(0, 0, 0), thickness=4, dashed=False):
        if style == "linear":
            if dashed:
                dashed_line(p1, p2, color=color, thickness=thickness)
            else:
                line(p1, p2, color=color, thickness=thickness)
            return

        def ease_out_expo(t):
            if t >= 1.0:
                return 1.0
            return 1 - np.power(2, -10 * t)

        pts = []
        for t in np.linspace(0, 1, n):
            eased = ease_out_expo(t)
            x = int(p1[0] + (p2[0] - p1[0]) * eased)
            y = int(p1[1] + (p2[1] - p1[1]) * t)
            pts.append((x, y))

        if dashed:
            dashed_polyline(pts, color=color, thickness=thickness)
        else:
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)

    def sample_neckline_curve(p1, p2, style="exponential", n=100):
        if style == "linear":
            return [p1, p2]

        def ease_out_expo(t):
            if t >= 1.0:
                return 1.0
            return 1 - np.power(2, -10 * t)

        pts = []
        for t in np.linspace(0, 1, n):
            eased = ease_out_expo(t)
            x = int(p1[0] + (p2[0] - p1[0]) * eased)
            y = int(p1[1] + (p2[1] - p1[1]) * t)
            pts.append((x, y))

        return pts

    def cubic_bezier(p1, ctrl1, ctrl2, p2, color=(0, 0, 0), thickness=4, n=120):
        pts = []

        for t in np.linspace(0, 1, n):
            x = (
                (1 - t) ** 3 * p1[0]
                + 3 * (1 - t) ** 2 * t * ctrl1[0]
                + 3 * (1 - t) * t ** 2 * ctrl2[0]
                + t ** 3 * p2[0]
            )
            y = (
                (1 - t) ** 3 * p1[1]
                + 3 * (1 - t) ** 2 * t * ctrl1[1]
                + 3 * (1 - t) * t ** 2 * ctrl2[1]
                + t ** 3 * p2[1]
            )
            pts.append([int(x), int(y)])

        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)

    def dashed_line(p1, p2, color=(150, 150, 150), thickness=2, dash=20, gap=12):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            return

        ux = dx / length
        uy = dy / length

        d = 0
        while d < length:
            start = (
                int(p1[0] + ux * d),
                int(p1[1] + uy * d)
            )

            end_d = min(d + dash, length)

            end = (
                int(p1[0] + ux * end_d),
                int(p1[1] + uy * end_d)
            )

            cv2.line(canvas, start, end, color, thickness, lineType=cv2.LINE_AA)
            d += dash + gap

    def dashed_polyline(points, color=(150, 150, 150), thickness=2, dash=20, gap=12):
        if len(points) < 2:
            return

        draw_dash = True
        remaining = dash

        for p1, p2 in zip(points[:-1], points[1:]):
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            segment_length = np.sqrt(dx**2 + dy**2)

            if segment_length == 0:
                continue

            ux = dx / segment_length
            uy = dy / segment_length
            travelled = 0

            while travelled < segment_length:
                step = min(remaining, segment_length - travelled)
                start = (
                    int(x1 + ux * travelled),
                    int(y1 + uy * travelled)
                )
                end = (
                    int(x1 + ux * (travelled + step)),
                    int(y1 + uy * (travelled + step))
                )

                if draw_dash:
                    cv2.line(canvas, start, end, color, thickness, lineType=cv2.LINE_AA)

                travelled += step
                remaining -= step

                if remaining <= 0:
                    draw_dash = not draw_dash
                    remaining = dash if draw_dash else gap

    # -------------------------------------------------
    # Contour-based influence
    # -------------------------------------------------
    right_profile = p.get("right_profile")
    right_neckline_profile = p.get("right_neckline_profile")
    right_strap_profile = p.get("right_strap_profile")
    valid_strap_profile = False

    if right_strap_profile is not None:
        profile_height = right_strap_profile.get("height_px") or 0
        profile_width = right_strap_profile.get("width_px") or 0
        profile_points = right_strap_profile.get("edge_points") or []
        valid_strap_profile = (
            profile_height >= 20 and
            profile_width >= 8 and
            len(profile_points) >= 8
        )

    has_detected_strap = (
        detected_strap_length is not None or
        valid_strap_profile
    )

    # These control how much the contour affects the final pattern.
    # Increase these if you want the image to affect the draft more.
    # Decrease these if the pattern becomes too wonky.
    waist_influence = 0.45
    chest_influence = 0.35
    top_influence = 0.30

    # Base pattern landmarks from measurements
    waist_x = ww
    chest_x = cw
    top_x = sw

    if right_profile is not None:
        waist_shape = right_profile.get("waist")
        chest_shape = right_profile.get("chest")
        top_shape = right_profile.get("top")

        if waist_shape is not None:
            # waist_shape[0] is normalized x position from contour
            # subtract 0.5 so it acts like a small left/right correction
            waist_x = ww + (waist_shape[0] - 0.5) * waist_influence * hw

        if chest_shape is not None:
            chest_x = cw + (chest_shape[0] - 0.5) * chest_influence * hw

        if top_shape is not None:
            top_x = sw + (top_shape[0] - 0.5) * top_influence * hw

    # Constrain points so they still look like a real pattern
    waist_x = np.clip(waist_x, hw * 0.45, hw * 1.05)
    chest_x = np.clip(chest_x, hw * 0.50, hw * 1.10)
    top_x = np.clip(top_x, hw * 0.35, hw * 1.10)

    neckline_depth = max(25, min(shoulder_length - chest_length, shoulder_length * 0.30))
    center_neck_length = shoulder_length - neckline_depth

    strap_lean = 0
    strap_top_width = strap_width

    if valid_strap_profile:
        lean_norm = right_strap_profile.get("lean_norm") or 0
        top_width_ratio = right_strap_profile.get("top_width_ratio") or 1

        strap_lean = np.clip(lean_norm * strap_width * 0.45, -strap_width * 0.65, strap_width * 0.65)
        strap_top_width = np.clip(strap_width * top_width_ratio, strap_width * 0.70, strap_width * 1.35)

    # -------------------------------------------------
    # Canvas origin and page sizing
    # -------------------------------------------------
    pattern_top_length = shoulder_length + strap_length if has_detected_strap else shoulder_length
    margin_mm = 20
    leftmost_x_mm = min(-sa, 0, hw, waist_x, chest_x, top_x, top_x + strap_lean - strap_top_width - sa)
    rightmost_x_mm = max(hw, waist_x, chest_x, top_x, top_x + sa, top_x + strap_lean + strap_top_width + sa)
    topmost_y_mm = pattern_top_length + sa
    bottommost_y_mm = -sa

    canvas_w = max(page_width, mm(rightmost_x_mm - leftmost_x_mm + margin_mm * 2))
    canvas_h = max(page_height, mm(topmost_y_mm - bottommost_y_mm + margin_mm * 2))
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    margin_px = mm(margin_mm)
    x0 = margin_px + mm(-leftmost_x_mm)
    y0 = canvas_h - margin_px

    def pt(x_mm, y_mm):
        return (
            x0 + mm(x_mm),
            y0 - mm(y_mm)
        )

    # -------------------------------------------------
    # Pattern landmarks
    # -------------------------------------------------
    hem_left = pt(0, 0)
    hem_right = pt(hw, 0)

    waist_pt = pt(waist_x, waist_length)
    chest_pt = pt(chest_x, chest_length)

    top_left = pt(0, center_neck_length)
    top_right = pt(top_x, shoulder_length)

    # Strap is based on top_right
    strap_outer_bot = top_right
    strap_outer_top = pt(top_x + strap_lean, shoulder_length + strap_length)

    strap_inner_bot = pt(top_x - strap_width, shoulder_length)
    strap_inner_top = pt(top_x + strap_lean - strap_top_width, shoulder_length + strap_length)

    if not has_detected_strap:
        shoulder_inset = min(strap_width, top_x * 0.35)
        strap_inner_bot = pt(top_x - shoulder_inset, shoulder_length)
        strap_inner_top = strap_inner_bot
        strap_outer_top = top_right

    points = [
    hem_left,
    hem_right,
    waist_pt,
    chest_pt,
    top_left,
    top_right,
    strap_inner_bot,
    strap_inner_top,
    strap_outer_top,
    ]

    # -------------------------------------------------
    # Draw main outline
    # -------------------------------------------------

    # Hem
    line(hem_left, hem_right)

    # Center front fold line
    dashed_line(hem_left, top_left, color=(120, 120, 120), thickness=3)

    # -------------------------------------------------
    # Clean side seam using contour-influenced landmarks
    # -------------------------------------------------

    # Curve controls are based on actual landmark positions.
    # This keeps the side seam smooth instead of copying the raw contour.
    ctrl_hem_waist_x = (hw + waist_x) / 2
    ctrl_hem_waist_y = waist_length / 2

    ctrl_waist_chest_x = (waist_x + chest_x) / 2
    ctrl_waist_chest_y = (waist_length + chest_length) / 2

    ctrl_chest_top_x = (chest_x + top_x) / 2
    ctrl_chest_top_y = (chest_length + shoulder_length) / 2

    # Add small shaping based on the waist indentation
    waist_indent = hw - waist_x

    ctrl_hem_waist_x -= waist_indent * 0.10
    ctrl_waist_chest_x -= waist_indent * 0.05
    ctrl_chest_top_x += waist_indent * 0.03

    quadratic_bezier(
        hem_right,
        pt(ctrl_hem_waist_x, ctrl_hem_waist_y),
        waist_pt
    )

    quadratic_bezier(
        waist_pt,
        pt(ctrl_waist_chest_x, ctrl_waist_chest_y),
        chest_pt
    )

    quadratic_bezier(
        chest_pt,
        pt(ctrl_chest_top_x, ctrl_chest_top_y),
        top_right
    )

    # Neckline
    neckline_span = max(top_x - strap_width, 1)
    neckline_scoop = neckline_depth * 0.18

    if right_neckline_profile is not None:
        neckline_scoop = neckline_depth * 0.30

    neck_ctrl = pt(neckline_span * 0.52, center_neck_length - neckline_scoop)

    neckline_curve_style = "linear" if neckline_style == "Straight (linear)" else "exponential"
    neckline_pts = sample_neckline_curve(top_left, strap_inner_bot, style=neckline_curve_style)

    if neckline_curve_style == "linear":
        draw_neckline_curve(top_left, strap_inner_bot, style="linear")
    else:
        draw_neckline_curve(top_left, strap_inner_bot, style="exponential")

    # Strap / shoulder fallback
    if has_detected_strap:
        line(strap_inner_bot, strap_inner_top)
        line(strap_inner_top, strap_outer_top)
        line(strap_outer_top, strap_outer_bot)
    else:
        line(strap_inner_bot, top_right)

    outline_points = [top_left]
    outline_points.extend(neckline_pts[1:])

    if has_detected_strap:
        outline_points.extend([
            strap_inner_top,
            strap_outer_top,
            strap_outer_bot
        ])
    else:
        outline_points.append(top_right)

    outline_points.extend(
        quadratic_points(top_right, pt(ctrl_chest_top_x, ctrl_chest_top_y), chest_pt)[1:]
    )
    outline_points.extend(
        quadratic_points(chest_pt, pt(ctrl_waist_chest_x, ctrl_waist_chest_y), waist_pt)[1:]
    )
    outline_points.extend(
        quadratic_points(hem_right, pt(ctrl_hem_waist_x, ctrl_hem_waist_y), waist_pt)[1:]
    )
    outline_points.append(hem_left)

    # -------------------------------------------------
    # Seam allowance guide
    # -------------------------------------------------
    sa_hem_left = pt(-sa, -sa)
    sa_hem_right = pt(hw + sa, -sa)

    sa_top_left = pt(-sa, shoulder_length + sa)
    pattern_top_length = shoulder_length + strap_length if has_detected_strap else shoulder_length
    sa_top_right = pt(top_x + sa, pattern_top_length + sa)

    dashed_line(sa_hem_left, sa_hem_right)

    sa_waist_pt = pt(waist_x + sa, waist_length)
    sa_chest_pt = pt(chest_x + sa, chest_length)
    sa_top_body = pt(top_x + sa, shoulder_length + sa)

    sa_neck_left = pt(-sa, center_neck_length + sa)
    sa_strap_inner_bot = pt(top_x - strap_width - sa, shoulder_length + sa)
    sa_strap_inner_top = pt(top_x + strap_lean - strap_top_width - sa, shoulder_length + strap_length + sa)
    sa_strap_outer_top = pt(top_x + strap_lean + sa, shoulder_length + strap_length + sa)

    # Draw a seam allowance line that follows the neckline and strap.
    dashed_line(sa_top_left, sa_neck_left)
    draw_neckline_curve(
        sa_neck_left,
        sa_strap_inner_bot,
        style="exponential",
        dashed=True
    )
    dashed_line(sa_strap_inner_bot, sa_strap_inner_top)
    dashed_line(sa_strap_inner_top, sa_strap_outer_top)
    if has_detected_strap:
        dashed_line(sa_strap_outer_top, sa_top_body)
    else:
        dashed_line(sa_strap_outer_top, sa_top_right)

    side_sa_points = []
    side_sa_points.extend(
        quadratic_points(
            sa_hem_right,
            pt(ctrl_hem_waist_x + sa, ctrl_hem_waist_y),
            sa_waist_pt
        )
    )
    side_sa_points.extend(
        quadratic_points(
            sa_waist_pt,
            pt(ctrl_waist_chest_x + sa, ctrl_waist_chest_y),
            sa_chest_pt
        )[1:]
    )
    side_sa_points.extend(
        quadratic_points(
            sa_chest_pt,
            pt(ctrl_chest_top_x + sa, ctrl_chest_top_y),
            sa_top_body
        )[1:]
    )

    if has_detected_strap:
        dashed_line(sa_top_body, sa_top_right)

    dashed_polyline(side_sa_points)
    dashed_line(sa_top_left, sa_hem_left)

    # -------------------------------------------------
    # Grain line
    # -------------------------------------------------
    grain_inset = max(20, min(hw * 0.18, top_x * 0.45))
    grain_x = x0 + mm(grain_inset)
    grain_top_y = min(center_neck_length, shoulder_length) * 0.88
    grain_bottom_y = max(35, side_length * 0.08)
    grain_top = (grain_x, y0 - mm(grain_top_y))
    grain_bottom = (grain_x, y0 - mm(grain_bottom_y))

    cv2.arrowedLine(
        canvas,
        grain_bottom,
        grain_top,
        (0, 0, 0),
        3,
        tipLength=0.08,
        line_type=cv2.LINE_AA
    )

    cv2.arrowedLine(
        canvas,
        grain_top,
        grain_bottom,
        (0, 0, 0),
        3,
        tipLength=0.08,
        line_type=cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "Grain Line",
        (grain_x + 10, grain_top[1] + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )

    # -------------------------------------------------
    # Labels
    # -------------------------------------------------
    cv2.putText(
        canvas,
        "Tank Top",
        pt(hw * 0.25, shoulder_length * 0.52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (60, 60, 60),
        2
    )

    cv2.putText(
        canvas,
        "Front Body",
        pt(hw * 0.25, shoulder_length * 0.47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (60, 60, 60),
        2
    )

    cv2.putText(
        canvas,
        "CUT ON FOLD",
        (x0 - 130, y0 - mm(shoulder_length * 0.45)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (60, 60, 60),
        1
    )

    label(hem_left, "hem", offset=(10, 35))
    label(top_left, "CF neck")

    if has_detected_strap:
        label(strap_inner_top, "strap top", offset=(-120, -10))
        label(strap_outer_top, "side strap", offset=(10, -10))

    # -------------------------------------------------
    # Measurement annotations
    # -------------------------------------------------
    cv2.putText(
        canvas,
        f"hem: {hem_width:.0f} mm",
        pt(hw * 0.35, -20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (90, 90, 160),
        1
    )

    cv2.putText(
        canvas,
        f"waist: {waist_width:.0f} mm",
        pt(waist_x + 10, waist_length),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (90, 90, 160),
        1
    )

    cv2.putText(
        canvas,
        f"chest: {chest_width:.0f} mm",
        pt(chest_x + 10, chest_length),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (90, 90, 160),
        1
    )

    cv2.putText(
        canvas,
        f"length: {body_length:.0f} mm",
        (x0 - 220, y0 - mm(shoulder_length * 0.5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (90, 90, 160),
        1
    )

    # -------------------------------------------------
    # Debug dots
    # -------------------------------------------------
    guide_points = [
        hem_left,
        hem_right,
        waist_pt,
        chest_pt,
        top_left,
        top_right,
        strap_inner_bot,
    ]

    if has_detected_strap:
        guide_points.extend([
            strap_inner_top,
            strap_outer_bot,
            strap_outer_top
        ])

    for point in guide_points:
        dot(point)

    # Optional: write whether right_profile was used
    profile_text = "profile: used" if right_profile is not None else "profile: not found"

    cv2.putText(
        canvas,
        profile_text,
        (x0 + 20, y0 - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (100, 100, 100),
        1
    )

    success = cv2.imwrite(output_path, canvas)
    print(f"saved: {success} → {output_path}")
    return canvas, outline_points


def choose_neckline_style(file_name):
    valid_styles = {
        "1": "Straight (linear)",
        "2": "Soft scoop (exponential)",
        "3": "Auto"
    }

    if not sys.stdin.isatty():
        print(f"[debug] stdin not interactive; defaulting neckline style to Auto")
        return "Auto"

    prompt = (
        f"Choose neckline style for {file_name}:\n"
        "  1) Straight (linear)\n"
        "  2) Soft scoop (exponential)\n"
        "  3) Auto\n"
        "Enter 1, 2, or 3 (default 3): "
    )

    choice = input(prompt).strip()
    return valid_styles.get(choice, "Auto")

def prepare_printing(image_name, canvas, output_folder, dpi=DEFAULT_DPI):
    """Slice a rendered pattern canvas into 8"x11" pages at the specified DPI."""

    print(f"[debug] prepare_printing dpi={dpi}")
    if canvas is None or canvas.size == 0:
        raise ValueError("prepare_printing() needs a valid canvas image")

    page_width = int(round(8 * dpi))
    page_height = int(round(11 * dpi))

    h, w = canvas.shape[:2]
    os.makedirs(output_folder, exist_ok=True)
    page_files = []

    if w <= page_width and h <= page_height:
        page_img = np.ones((page_height, page_width, 3), dtype=np.uint8) * 255
        x_offset = (page_width - w) // 2
        y_offset = (page_height - h) // 2
        page_img[y_offset:y_offset + h, x_offset:x_offset + w] = canvas

        cv2.putText(
            page_img,
            "Page 1",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 0),
            4,
            lineType=cv2.LINE_AA
        )

        filename = os.path.join(output_folder, f"{image_name}_pattern_page_01.png")
        cv2.imwrite(filename, page_img)
        print("Saved", filename)
        page_files.append(filename)
        return page_files

    num_columns = int(np.ceil(w / page_width))
    num_rows = int(np.ceil(h / page_height))
    page_number = 1

    for row in range(num_rows):
        for col in range(num_columns):
            x0 = col * page_width
            y0 = row * page_height
            x1 = min(x0 + page_width, w)
            y1 = min(y0 + page_height, h)

            page_img = np.ones((page_height, page_width, 3), dtype=np.uint8) * 255
            crop = canvas[y0:y1, x0:x1]
            page_img[0:crop.shape[0], 0:crop.shape[1]] = crop

            page_label = f"Page {page_number}"
            cv2.putText(
                page_img,
                page_label,
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 0),
                4,
                lineType=cv2.LINE_AA
            )

            filename = os.path.join(output_folder, f"{image_name}_pattern_page_{page_number:02d}.png")
            cv2.imwrite(filename, page_img)
            print("Saved", filename)
            page_files.append(filename)
            page_number += 1

    return page_files

def main():
    folder_path   = Path('/Users/haileyvan/Downloads/raw_shirt/input')
    output_folder = "/Users/haileyvan/Downloads/raw_shirt/output"
    os.makedirs(output_folder, exist_ok=True)

    for file in folder_path.iterdir():
        if file.name.startswith('.'):
            continue
        if file.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue

        output_path = os.path.join(output_folder, file.stem + "_exported.png")
        output_path2 = os.path.join(output_folder, file.stem + "_pattern.png")
        label_path  = os.path.join(output_folder, file.stem + ".json")
        image_path  = str(file)
        image  = cv2.imread(image_path)
        segmented_mask = segment_garment_only(image_path, output_path)

        try:
            binary = clean_mask(segmented_mask, flatten_top=False)
            outline = find_outline(binary, image, output_path)
            measurements, pts = get_bounding_measurements(outline)
            chest_y = measurements["image_y"].get("chest_y_px")
            shoulder_y = measurements["image_y"].get("shoulder_y_px")
            right_neckline_profile = extract_clean_neckline_profile(
                outline,
                chest_y=chest_y,
                shoulder_y=shoulder_y,
                side="right"
            )
            right_strap_profile = extract_clean_strap_profile(outline, side="right")
            print("using natural neckline profile")
        except ValueError as e:
            print(f"natural neckline failed, using 20% top cleanup: {e}")
            binary = clean_mask(segmented_mask, flatten_top=True)
            outline = find_outline(binary, image, output_path)
            measurements, pts = get_bounding_measurements(outline)
            right_neckline_profile = None
            right_strap_profile = None

        # Save cleaned mask to inspect
        cv2.imwrite(output_path.replace(".png", "_mask_clean.png"), binary)

        right_profile = extract_clean_side_profile(outline, side="right")
        measurements["right_profile"] = right_profile
        measurements["right_neckline_profile"] = right_neckline_profile
        measurements["right_strap_profile"] = right_strap_profile

        measurements_mm = px_to_mm(measurements)
        measurements_mm["right_profile"] = right_profile
        measurements_mm["right_neckline_profile"] = right_neckline_profile
        measurements_mm["right_strap_profile"] = right_strap_profile

        neckline_style = choose_neckline_style(file.name)
        canvas, outline_points = draft_pattern(
            measurements_mm,
            output_path2,
            neckline_style=neckline_style,
            dpi=DEFAULT_DPI
        )

        save_labels(measurements_mm, label_path)
        print(f"{file.name}: {measurements_mm}")
        prepare_printing(file.stem, canvas, output_folder, dpi=DEFAULT_DPI)



if __name__ == "__main__":
    main()
