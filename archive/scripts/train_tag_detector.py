from ultralytics import YOLO

DATA_YAML = "datasets/bee_tag/bee-tag-detection.v1i.yolov8/data.yaml" # change DATA_YAML to the new dataset

def main():
    model = YOLO("runs/tag_detection/train/weights/best.pt") # continue where our YOLO model left off

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0, # GPU 
        project="runs/tag_detection",
        name="train"
    )

if __name__ == "__main__":
    main()
