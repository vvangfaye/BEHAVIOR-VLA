"""Compare RGB-D button detections with annotations from the train-only capture."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from radio_policy import CAMERA_POSES_KEY, HEAD_KEY, HeadCamera, find_button, find_housing


def audit(directory):
    results = []
    for line in (directory / "frames.jsonl").read_text().splitlines():
        row = json.loads(line)
        if not 0 <= row["id"] < 300:
            raise ValueError("Capture contains an instance outside the development split")
        basename = directory / row["stem"]
        rgb = cv2.cvtColor(cv2.imread(str(basename.with_suffix(".jpg"))), cv2.COLOR_BGR2RGB)
        with np.load(basename.with_suffix(".npz"), allow_pickle=False) as frame:
            obs = {HEAD_KEY + "::rgb": rgb, HEAD_KEY + "::depth_linear": frame["depth"],
                   CAMERA_POSES_KEY: np.tile(frame["camera_pose"], 3)}
            camera = HeadCamera(obs)
            housing = find_housing(camera)
            prediction = find_button(camera, housing=housing) if housing is not None else None
        projection = row["projected_marker_uv_depth"]
        depth = row["sampled_depth"]
        visible = projection is not None and depth is not None and abs(projection[2] - depth) <= 0.055
        error = None
        if visible and prediction is not None:
            error = float(np.linalg.norm(np.asarray(prediction.uv) - projection[:2]))
        results.append({"instance": row["id"], "step": row["step"], "marker_visible": visible, "housing": housing is not None,
                        "predicted_uv": None if prediction is None else prediction.uv,
                        "projected_uv": None if projection is None else projection[:2], "pixel_error": error})
    return {"frames": len(results), "visible": sum(row["marker_visible"] for row in results),
            "matched_within_10_px": sum(row["pixel_error"] is not None and row["pixel_error"] <= 10 for row in results),
            "false_positives": sum(not row["marker_visible"] and row["predicted_uv"] is not None for row in results),
            "per_frame": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.capture), indent=2))
