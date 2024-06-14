import argparse
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=str)
    args = parser.parse_args()

    ANNOTATIONS_DIR = args.annotations
    STATS_DIR = os.path.join(ANNOTATIONS_DIR, "stats")
    OUT_FILE = os.path.join(ANNOTATIONS_DIR, "gt.txt")

    id_offset = 1
    TIME_SEP = 10
    prev_frame = None
    max_count = 0

    trackers = []

    for stat_filename in os.listdir(STATS_DIR):
        stats = np.load(os.path.join(STATS_DIR, stat_filename))

        frame = int(stat_filename.split(".")[0].split("_")[-1])
        if prev_frame != None:
            if frame - prev_frame > TIME_SEP:
                id_offset += max_count
                max_count = 0

        if stats.shape[0] > max_count:
            max_count = stats.shape[0]

        for i, stat in enumerate(stats):
            trackers.append(
                ", ".join(
                    map(
                        str,
                        [
                            frame,
                            i + id_offset,
                            stat[0],
                            stat[1],
                            stat[2],
                            stat[3],
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                    )
                )
                + "\n"
            )

        prev_frame = frame

    with open(OUT_FILE, "w") as f:
        f.writelines(trackers)
