import json
from pathlib import Path
from datetime import datetime
import uuid
from analyze_helpers import (
    analyze_annotations,
    analyze_camera_statistics,
    analyze_lidar_statistics
)

def main():
    # Hardcoded arguments
    dataset_dir = Path("waymo_dataset")
    output = Path("integrations/waymo_analysis.json")
    frame_list = "splits_clft/all.txt"
    components = ["annotations", "camera", "lidar"]
    workers = 8

    # Class mapping for Waymo
    class_names = {
        0: "ignore",
        1: "vehicle",
        2: "pedestrian",
        3: "sign",
        4: "cyclist",
        5: "background"
    }

    # Read frame list
    frame_list_file = dataset_dir / frame_list
    if not frame_list_file.exists():
        print(f"Frame list file {frame_list_file} does not exist")
        return

    with open(frame_list_file, 'r') as f:
        frame_lines = f.read().splitlines()

    # Collect file paths
    annotation_files = []
    camera_files = []
    lidar_files = []

    for line in frame_lines:
        if line.strip():
            # Assuming line is like labeled/day/not_rain/camera/segment-xxx.png
            parts = line.split('/')
            if len(parts) >= 4 and parts[3] == 'camera':
                base_path = '/'.join(parts[:-2])  # labeled/day/not_rain
                segment_name = parts[-1].replace('.png', '')  # segment-xxx
                
                annotation_path = dataset_dir / base_path / "annotation" / f"{segment_name}.png"
                camera_path = dataset_dir / line
                lidar_path = dataset_dir / base_path / "lidar" / f"{segment_name}.pkl"
                
                if annotation_path.exists():
                    annotation_files.append(annotation_path)
                if camera_path.exists():
                    camera_files.append(camera_path)
                if lidar_path.exists():
                    lidar_files.append(lidar_path)

    results = {
        "uuid": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "dataset_info": {
            "path": str(dataset_dir),
            "frame_list_file": frame_list,
            "total_frames_in_list": len(frame_lines)
        }
    }

    # Analyze components
    if "annotations" in components:
        results["annotations"] = analyze_annotations(annotation_files, class_names, workers)

    if "camera" in components:
        results["camera"] = analyze_camera_statistics(camera_files, workers)

    if "lidar" in components:
        results["lidar"] = analyze_lidar_statistics(lidar_files, workers)

    # Save results
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
