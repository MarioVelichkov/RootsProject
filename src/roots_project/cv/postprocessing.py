from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize


def clean_root_mask(
    prediction: np.ndarray,
    threshold: float,
    min_size: int,
    kernel_size: int,
    dilate_iterations: int,
    close_iterations: int,
) -> np.ndarray:
    binary = prediction > threshold
    cleaned = remove_small_objects(binary, min_size=min_size).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(cleaned, kernel, iterations=dilate_iterations)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return (closed > 0).astype(np.uint8)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask > 0)


@dataclass(frozen=True)
class _TipCandidate:
    row: float
    col: float
    skeleton_length: float


def extract_root_tips(
    skeleton: np.ndarray,
    max_roots: int,
    plant_merge_distance_px: float = 128.0,
) -> list[tuple[float, float]]:
    from skan import Skeleton, summarize

    if not np.any(skeleton):
        return []

    skeleton_data = Skeleton(skeleton)
    branch_data = summarize(skeleton_data)
    if branch_data.empty:
        return []

    candidates: list[_TipCandidate] = []
    for skeleton_id, group in branch_data.groupby("skeleton-id"):
        skeleton_length = float(group["branch-distance"].sum())
        row_max_index = group[["image-coord-src-0", "image-coord-dst-0"]].max(axis=1).idxmax()
        branch = group.loc[row_max_index]

        if branch["image-coord-src-0"] >= branch["image-coord-dst-0"]:
            row = branch["image-coord-src-0"]
            col = branch["image-coord-src-1"]
        else:
            row = branch["image-coord-dst-0"]
            col = branch["image-coord-dst-1"]
        candidates.append(_TipCandidate(row=float(row), col=float(col), skeleton_length=skeleton_length))

    candidates.sort(key=lambda candidate: candidate.col)

    plant_groups: list[list[_TipCandidate]] = []
    for candidate in candidates:
        if not plant_groups or candidate.col - plant_groups[-1][-1].col > plant_merge_distance_px:
            plant_groups.append([candidate])
        else:
            plant_groups[-1].append(candidate)

    plant_tips: list[tuple[_TipCandidate, float]] = []
    for group in plant_groups:
        tip = max(group, key=lambda candidate: (candidate.row, candidate.skeleton_length))
        confidence = sum(candidate.skeleton_length for candidate in group)
        plant_tips.append((tip, confidence))

    plant_tips.sort(key=lambda item: item[1], reverse=True)
    endpoints = [(tip.row, tip.col) for tip, _ in plant_tips[:max_roots]]

    return endpoints
