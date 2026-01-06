# Bee Project

This project is an academic, deep learning system
for analyzing bee behavior from video data.

The project is designed as a modular pipeline, where each stage focuses on a different
aspect of bee analysis. The current implementation represents the first stage of the project,
with additional models and analysis components planned for future stages.

---

## Project Stages Overview

### Stage 1: Bee Tag Detection and Number Recognition (Current)
- Detection of bee tags from video frames using a custom YOLOv8 object detection model.
- Extraction of the numeric identifier printed on each tag using OCR techniques.
- This stage enables reliable identification and tracking of individual bees.

### Planned Future Stages
- Bee dance detection and classification.

The project is structured to allow easy extension by adding new models and processing
stages without modifying the existing pipeline.

---

## Environment Setup

It is recommended to work inside a virtual environment.

1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

---

## Commands and Usage

A list of commonly used commands for data preparation, training, and inference
is available in the following file: docs/commands.txt

