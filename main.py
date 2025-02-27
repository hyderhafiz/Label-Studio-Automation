import os
import json
import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Import the conversion function from rlepoly.py
from rlepoly import convert_to_yolo

# We'll also need to import your YOLO model for training events.
from model import YOLO

# Define directories
base_dir = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(base_dir, "data/images")
RESPONSE_DIR = os.path.join(base_dir, "data/response")
TXT_DIR = os.path.join(base_dir, "data/labels")# destination for YOLO TXT files

# Ensure directories exist
os.makedirs(RESPONSE_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

app = FastAPI(
    title="Label Studio ML Backend",
    description="A minimal FastAPI app to work as an ML backend for Label Studio"
)

@app.get("/")
def root():
    return {"Hello": "World"}

@app.get("/health")
def health():
    return JSONResponse(status_code=200, content={"status": "UP"})

@app.post("/setup")
async def setup_endpoint(request: Request):
    # Optional: Process setup if needed
    data = await request.json()
    return JSONResponse(status_code=200, content={"model_version": "yolo_backend_v1", "status": "Setup successful"})

@app.post("/webhook")
async def webhook(request: Request):
    """
    This endpoint handles both annotation events and training triggers.
    - For annotation events (e.g., ANNOTATION_CREATED), it converts the JSON into a YOLO TXT file.
    - For a training trigger (START_TRAINING), it calls the YOLO model's fit() method.
    """
    payload = await request.json()
    action = payload.get("action")
    logger = logging.getLogger(__name__)
    logger.info(f"Received webhook event: {action}")

    # Save the payload to a file for record-keeping.
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    annotation_id = payload.get("annotation", {}).get("id", "noid")
    filename = f"response_{annotation_id}_{timestamp}.json"
    filepath = os.path.join(RESPONSE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    logger.info(f"Saved payload to {filepath}")

    if action == "START_TRAINING":
        # When training is triggered, we call our YOLO model's fit() method.
        try:
            # Instantiate and set up the YOLO model.
            yolo_model = YOLO()
            yolo_model.setup()
            # Pass the training event and payload (or an empty dict) to fit()
            result = yolo_model.fit(action, payload)
            logger.info("Training complete with result: %s", result)
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    else:
        # For annotation events, try converting the annotation.
        # Check if the payload has a "task" field containing image info.
        if not payload.get("task", {}).get("data", {}).get("image"):
            logger.warning("No image found in task data.")
        else:
            try:
                convert_to_yolo(filepath, IMAGES_DIR)
                logger.info("Converted JSON to YOLO TXT file.")
            except Exception as e:
                logger.error(f"Error during conversion: {e}")

        image_path = payload.get("task", {}).get("data", {}).get("image")
        if not image_path:
            logger.warning("No image found in task data.")
        else:
            save_image_url(image_path)  # Save the image URL to the file


        # Process annotation event details
        if payload.get("action") == "ANNOTATION_CREATED":
            annotation = payload.get("annotation", {})
            annotation_id = annotation.get("id")
            task_id = annotation.get("task")
            lead_time = annotation.get("lead_time")
            completed_by = annotation.get("completed_by")
            logger.info(f"Annotation Created: ID: {annotation_id}, Task: {task_id}, Lead Time: {lead_time}, Completed By: {completed_by}")
            return JSONResponse(
                status_code=200,
                content={"status": "success", "message": f"Annotation {annotation_id} processed"}
            )
        else:
            logger.info(f"Unhandled event type: {action}")
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "message": "Event received but not specifically handled"}
            )


def save_image_url(image_path: str):
    """
    Save the image name and URL in a text file without duplicates.
    """
    txt_file_path = os.path.join(base_dir, "data/imageurl.txt")
    
    # Extract image name and create a formatted entry
    image_name = os.path.basename(image_path)  # Extracts filename from path
    image_entry = f"{image_name} {image_path}\n"

    # Check if the entry already exists
    if os.path.exists(txt_file_path):
        with open(txt_file_path, "r") as file:
            existing_lines = file.readlines()
        if image_entry in existing_lines:
            return  # Skip duplicate entries

    # Append the new image entry
    with open(txt_file_path, "a") as file:
        file.write(image_entry)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
