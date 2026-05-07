from ultralytics import YOLO

DATA_YAML = "archive/datasets/bee_tag/bee-tag-detection.v4i.yolov8/data.yaml" # change DATA_YAML to the new dataset

def main():
    model = YOLO("backend/models/tag_detection/best.pt") # continue where our YOLO model left off

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0, # GPU 
        patience=15,
        project="runs/tag_detection",
        name="train"
    )

if __name__ == "__main__":
    main()
