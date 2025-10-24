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
    dataset_dir = Path("zod_dataset")
    output = Path("integrations/zod_analysis.json")
    frame_list = "all.txt"
    components = ["annotations", "camera", "lidar"]
    workers = 8

    # Class mapping from SAM processing
    class_names = {
        0: "background",
        1: "ignore",
        2: "vehicle",
        3: "sign",
        4: "cyclist",
        5: "pedestrian"
    }

    # Collect file paths
    annotation_files = []
    camera_files = []
    lidar_files = []

    if "annotations" in components or "all" in components:
        annotation_dir = dataset_dir / "annotation"
        if annotation_dir.exists():
            annotation_files = list(annotation_dir.glob("frame_*.png"))
        else:
            print(f"Annotation directory {annotation_dir} not found")

    if "camera" in components or "all" in components:
        camera_dir = dataset_dir / "camera"
        if camera_dir.exists():
            camera_files = list(camera_dir.glob("frame_*.png"))
        else:
            print(f"Camera directory {camera_dir} not found")

    if "lidar" in components or "all" in components:
        lidar_dir = dataset_dir / "lidar"
        if lidar_dir.exists():
            lidar_files = list(lidar_dir.glob("frame_*.pkl"))
        else:
            print(f"LiDAR directory {lidar_dir} not found")

    results = {
        "uuid": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "dataset_info": {
            "path": str(dataset_dir),
            "frame_list_file": frame_list,
            "total_frames_in_list": len(annotation_files) if annotation_files else 0
        }
    }

    # Analyze components
    if annotation_files:
        results["annotations"] = analyze_annotations(annotation_files, class_names, workers)

    if camera_files:
        results["camera"] = analyze_camera_statistics(camera_files, workers)

    if lidar_files:
        results["lidar"] = analyze_lidar_statistics(lidar_files, workers)

    # Save results
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
