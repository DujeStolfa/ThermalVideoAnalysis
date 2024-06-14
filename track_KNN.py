"""
Praćenje sitnih objekata metodom KNN.
"""

import argparse
import os

import cv2
import faiss
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

DATASET_PATH = os.path.join("data", "01_cut_raw", "out")
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

ANOMALY_SCORE_THRESHOLDS = [600, 800, 1200, 1600, 2000, 2400]

K_FACTOR = 16
GATE_RADIUS = 60
T_LIMIT = 3
ALPHA = 0.95
BETA = 0.3


for thresh in ANOMALY_SCORE_THRESHOLDS:
    path = os.path.join(TRACKER_PATH, str(thresh))
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(TRACKER_PATH, str(thresh), TRACKER_FILENAME), "w") as f:
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

# Load data in batches of size `BCKG_WINDOW_SIZE`
window_cutoffs = list(
    range(0, int(total_frame_count - 2 * WINDOW_SIZE / 3), WINDOW_SIZE)
)
window_cutoffs.append(total_frame_count)

frame_mask = plt.imread(FRAME_MASK_FILE) / 255
frame_mask = frame_mask[:, :, np.newaxis].astype(np.uint).repeat(3, axis=2)

track_counts = [0 for _ in ANOMALY_SCORE_THRESHOLDS]

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
            frames.append(frame[..., ::-1])

        seek += 1

    cap.release()

    frames = np.array(frames) * frame_mask
    frames_count, frame_height, frame_width, depth = frames.shape

    # Calculate anomaly scores
    anomaly_scores = np.zeros((frame_height, frame_width, frames_count))

    for y in tqdm(range(frame_height), desc="\\ Calculating anomaly scores"):
        for x in range(frame_width):
            selected_pixel = frames[:, y, x, :]

            index = faiss.IndexFlatL2(depth)
            index.add(selected_pixel)
            neighbours, _ = index.search(selected_pixel, K_FACTOR)
            anomaly_scores[y][x] = np.mean(neighbours, axis=1)

    for thresh_idx, threshold in enumerate(ANOMALY_SCORE_THRESHOLDS):
        print(f"- Anomaly score threshold = {threshold}")
        masks = np.where(np.moveaxis(anomaly_scores, 2, 0) > threshold, 1, 0).astype(
            np.uint8
        )

        # Perceptual grouping
        print("\\ Running perceptual grouping")
        detections = []

        for frame_idx in range(masks.shape[0]):
            (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
                masks[frame_idx], connectivity=4, ltype=cv2.CV_32S
            )

            selected_labels = list(range(1, numLabels))

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
                    track_id = track_counts[thresh_idx] + 1
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
                    track_counts[thresh_idx] += 1

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

        with open(
            os.path.join(TRACKER_PATH, str(threshold), TRACKER_FILENAME), "a"
        ) as f:
            f.writelines(lines)

        print()
