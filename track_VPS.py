"""
Praćenje sitnih objekata metodom VPS
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("video", type=str)
parser.add_argument("extension", type=str)
args = parser.parse_args()


VIDEO = args.video
TRACKER_FILENAME = f"BAT_{VIDEO}.txt"
VIDEO_FILENAME = f"video_{VIDEO}.{args.extension}"

VARIANT = "BASE"

if args.extension == "mp4":
    DATASET_PATH = os.path.join("data", "01_cut_raw", "out")
else:
    DATASET_PATH = os.path.join("data", "00_raw", "video")
FRAME_MASK_FILE = os.path.join("data", "frame_mask.tif")
TIMESTAMP = datetime.now().strftime("%m%d-%H%M")
TRACKER_PATH = os.path.join(
    "TrackEval",
    "data",
    "trackers",
    "mot_challenge",
    f"BAT_{VIDEO}",
    f"track-{TIMESTAMP}",
    "data",
)

START_WINDOW_IDX = 1  # >= 1
WINDOW_SIZE = 128
BCKG_WINDOW_SIZE = 1024

BCKG_FACTOR = 8
BLOB_COUNT_LIMIT = 8

GATE_RADIUS = 20
T_LIMIT = 3
ALPHA = 0.95
BETA = 0.3


path = os.path.join(TRACKER_PATH, VARIANT)
if not os.path.exists(path):
    os.makedirs(path)

with open(os.path.join(TRACKER_PATH, VARIANT, TRACKER_FILENAME), "w") as f:
    f.write("")


# Count frames in video
print(os.path.join(DATASET_PATH, VIDEO_FILENAME))
cap = cv2.VideoCapture(os.path.join(DATASET_PATH, VIDEO_FILENAME))
total_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frame_count += 1

cap.release()
print(total_frame_count)

# Load data in batches of size `WINDOW_SIZE`
window_cutoffs = list(
    range(0, int(total_frame_count - 2 * WINDOW_SIZE / 3), WINDOW_SIZE)
)
window_cutoffs.append(total_frame_count)

# Calculate stats on windows of size `BCKG_WINDOW_STATS`
bckg_window_cutoffs = list(
    range(0, int(total_frame_count - 2 * BCKG_WINDOW_SIZE / 3), BCKG_WINDOW_SIZE)
)
bckg_window_cutoffs.append(total_frame_count)

bckg_window_stats = []
map_window_to_bckg = []
curr_cutoff_idx = 0

for bckg_window_idx in tqdm(
    range(1, len(bckg_window_cutoffs)),
    total=len(bckg_window_cutoffs) - 1,
    desc="\\Calculating stats",
):
    frames = []

    # Load window
    cap = cv2.VideoCapture(os.path.join(DATASET_PATH, VIDEO_FILENAME))
    seek = 0

    while seek < bckg_window_cutoffs[bckg_window_idx]:
        ret, frame = cap.read()

        if not ret:
            break

        if seek >= bckg_window_cutoffs[bckg_window_idx - 1]:
            frames.append(np.sum(frame, axis=-1) / 3)  # rgb to gray

        seek += 1

    cap.release()
    frames = np.array(frames)

    # Calculate stats
    mean = np.mean(frames, axis=0)
    std = np.std(frames, axis=0)

    bckg_window_stats.append((mean, std))

    # Set references for each subwindow
    while window_cutoffs[curr_cutoff_idx + 1] <= bckg_window_cutoffs[bckg_window_idx]:
        map_window_to_bckg.append(bckg_window_idx - 1)
        curr_cutoff_idx += 1

        if curr_cutoff_idx >= len(window_cutoffs) - 1:
            break

# Mask out parts of frame
frame_mask = plt.imread(FRAME_MASK_FILE) / 255
frame_mask = frame_mask.astype(np.uint)

track_count = 0

for window_idx in range(START_WINDOW_IDX, len(window_cutoffs)):
    print(f"Analysing window {window_idx} / {len(window_cutoffs) - 1}")
    curr_window_size = window_cutoffs[window_idx] - window_cutoffs[window_idx - 1]
    frames = []

    # Load window
    cap = cv2.VideoCapture(os.path.join(DATASET_PATH, VIDEO_FILENAME))
    seek = 0

    while seek < window_cutoffs[window_idx]:
        ret, frame = cap.read()

        if not ret:
            break

        if seek >= window_cutoffs[window_idx - 1]:
            frames.append(np.sum(frame, axis=-1) / 3)  # rgb to gray

        seek += 1

    cap.release()

    frames = np.array(frames) * frame_mask
    frames_count, frame_height, frame_width = frames.shape

    # Calculate segmentation masks
    vps_image = np.max(frames, axis=0)
    vps_frame_matrix = np.argmax(frames, axis=0)

    curr_mean, curr_std = bckg_window_stats[map_window_to_bckg[window_idx - 1]]
    vps_mask = np.where(vps_image > curr_mean + BCKG_FACTOR * curr_std, 1, 0)

    split_vps_masks = [
        np.where(vps_frame_matrix == i, 1, 0) for i in range(frames_count)
    ]
    split_vps_masks *= vps_mask

    # Mask pixels that significantly deviate from the background model
    masks = np.where(frames > curr_mean + BCKG_FACTOR * curr_std, 1, 0).astype(np.uint8)

    # Perceptual grouping
    print("\\ Running perceptual grouping")
    detections = []

    for frame_idx in range(masks.shape[0]):
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
            masks[frame_idx], connectivity=4, ltype=cv2.CV_32S
        )

        # Only select blobs that contain seed pixels
        masked_labels = labels * masks[frame_idx]
        unsorted_labels = np.unique(masked_labels)[1:]

        blobs_per_frame = min(numLabels, BLOB_COUNT_LIMIT)
        selected_labels = unsorted_labels[
            stats[unsorted_labels][:, cv2.CC_STAT_AREA].argsort()
        ][:-blobs_per_frame:-1]

        detections.append(
            list(
                zip(
                    centroids[selected_labels],
                    stats[selected_labels][:, cv2.CC_STAT_AREA],
                    stats[selected_labels][:, : cv2.CC_STAT_AREA],
                    selected_labels,
                )
            )
        )

    # Tracking
    print("\\ Tracking")
    active_tracks = dict()  # track_id: (track_length, tracklet_index)
    inactive_tracks = dict()
    tentative_tracks = dict()
    terminated_tracks = dict()
    state_vectors = []  # (frame, track_id, state)
    tracks = []  # frame, track_id, left, top, width, height, -1, -1, -1, -1

    for frame_idx, candidates in tqdm(
        enumerate(detections), total=len(detections), desc="\\ Tracking"
    ):
        frame = window_cutoffs[window_idx - 1] + frame_idx  # + 1
        assigned_blob_indices = []

        sorted_items_active = sorted(
            active_tracks.items(), key=lambda x: x[1][0], reverse=True
        )
        sorted_items_tentative = sorted(
            tentative_tracks.items(), key=lambda x: x[1][0], reverse=True
        )

        for track_id, (track_length, tracklet_index) in (
            sorted_items_active + sorted_items_tentative
        ):
            prev_frame, _, prev_state = state_vectors[tracklet_index]

            delta_t = frame - prev_frame
            K = np.array(
                [[ALPHA, 0], [0, ALPHA], [BETA / delta_t, 0], [0, BETA / delta_t]]
            )
            H = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ]
            )
            F = np.array(
                [
                    [1, 0, delta_t, 0],
                    [0, 1, 0, delta_t],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            extrapolated = np.dot(F, prev_state)

            min_d = None
            min_i = None
            for blob_idx, (cent, _, bbox, _) in enumerate(candidates):
                d = np.linalg.norm(extrapolated[:2] - cent)

                if (
                    d < GATE_RADIUS
                    and (min_d == None or min_d > d)
                    and blob_idx not in assigned_blob_indices
                ):
                    min_d = d
                    min_i = blob_idx

            if min_i != None:
                measurement, _, bbox, _ = candidates[min_i]

                innovation = measurement - np.dot(H, prev_state)
                new_state = prev_state + np.dot(K, innovation)
                tracklet = np.array(
                    [
                        frame,
                        track_id,
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                        -1,
                        -1,
                        -1,
                        -1,
                    ]
                )

                if track_id in active_tracks:
                    active_tracks[track_id] = (track_length + 1, len(state_vectors))
                else:
                    tentative_tracks[track_id] = (
                        track_length + 1,
                        len(state_vectors),
                    )
                state_vectors.append((frame, track_id, new_state))
                tracks.append(tracklet)

                assigned_blob_indices.append(min_i)

        # Track maintenance
        to_inactive = []
        for track_id, (track_length, tracklet_index) in active_tracks.items():
            if frame - state_vectors[tracklet_index][0] > T_LIMIT:
                to_inactive.append(track_id)

        for track_id in to_inactive:
            inactive_tracks[track_id] = active_tracks.pop(track_id)

        to_active = []
        to_terminated = []
        for track_id, (track_length, tracklet_index) in tentative_tracks.items():
            if track_length >= T_LIMIT:
                to_active.append(track_id)
            elif frame - state_vectors[tracklet_index][0] > T_LIMIT:
                to_terminated.append(track_id)

        for track_id in to_active:
            active_tracks[track_id] = tentative_tracks.pop(track_id)
        for track_id in to_terminated:
            terminated_tracks[track_id] = tentative_tracks.pop(track_id)

        for blob_idx, (centroid, _, bbox, _) in enumerate(candidates):
            if blob_idx not in assigned_blob_indices:
                track_id = track_count + 1
                state = np.array([centroid[0], centroid[1], 0, 0])
                tracklet = np.array(
                    [
                        frame,
                        track_id,
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                        -1,
                        -1,
                        -1,
                        -1,
                    ]
                )

                tentative_tracks[track_id] = (1, len(state_vectors))
                state_vectors.append((frame, track_id, state))
                tracks.append(tracklet)
                track_count += 1

    print(f"\\ Tracking results:")
    data = [
        ["Active", "Inactive", "Tentative", "Terminated"],
        [
            len(active_tracks),
            len(inactive_tracks),
            len(tentative_tracks),
            len(terminated_tracks),
        ],
    ]
    for row in data:
        fmt_str = ". " + "{:>13} " * 4
        print(fmt_str.format(*row))

    # Save tracker file
    lines = [
        ", ".join(map(str, x)) + "\n"
        for x in filter(
            lambda x: x[1] in active_tracks or x[1] in inactive_tracks, tracks
        )
    ]

    with open(os.path.join(TRACKER_PATH, VARIANT, TRACKER_FILENAME), "a") as f:
        f.writelines(lines)

    print()
