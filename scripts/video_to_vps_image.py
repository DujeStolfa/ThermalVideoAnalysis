import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("subfolder", type=str)
    parser.add_argument("background", type=int)
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    SUBFOLDER = args.subfolder
    BCKG_WINDOW_SIZE = args.background

    for filename in os.listdir(os.path.join(DATASET_PATH, SUBFOLDER)):
        # Count frames in video
        cap = cv2.VideoCapture(os.path.join(DATASET_PATH, SUBFOLDER, filename))
        total_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frame_count += 1

        cap.release()

        # Load data in batches of size `BCKG_WINDOW_SIZE`
        cap = cv2.VideoCapture(os.path.join(DATASET_PATH, SUBFOLDER, filename))
        bckg_cutoffs = list(
            range(
                0, int(total_frame_count - 2 * BCKG_WINDOW_SIZE / 3), BCKG_WINDOW_SIZE
            )
        )
        bckg_cutoffs.append(total_frame_count)

        out_dir = os.path.join(DATASET_PATH, "vps_images", filename.split(".")[0])
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        rg = list(range(1, len(bckg_cutoffs)))
        for bckg_window_idx in tqdm(rg, total=len(rg), desc=filename):
            curr_window_size = (
                bckg_cutoffs[bckg_window_idx] - bckg_cutoffs[bckg_window_idx - 1]
            )
            all_frames = []

            for i in range(curr_window_size):
                ret, frame = cap.read()

                if not ret:
                    break

                all_frames.append(frame[..., ::-1])

            all_frames = np.array(all_frames)

            global_vps_image = np.max(
                all_frames, axis=0
            )  # Only used for track visualisation

            out_index = "000" + str(bckg_window_idx)
            plt.imsave(
                os.path.join(
                    out_dir, filename.split(".")[0] + "_" + out_index[-3:] + ".png"
                ),
                global_vps_image,
            )
