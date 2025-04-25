import os
import json
import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from rlepoly import convert_to_yolo  # Import YOLO conversion script
from model import YOLO  # Import YOLO training class

# Define base directories
base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(base_dir, "data")
JSON_DIR = os.path.join(DATA_DIR, "json")         #  Central folder for all responses
PROJECTS_DIR = os.path.join(DATA_DIR, "projects") #  Each project has its own folder
CONFIG_FILE = os.path.join(DATA_DIR, "projects_config.json")  # Config file for active projects

# Ensure base directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(PROJECTS_DIR, exist_ok=True)

app = FastAPI(
    title="Label Studio ML Backend",
    description="A minimal FastAPI app to work as an ML backend for Label Studio"
)

@app.get("/health")
def health():
    return JSONResponse(status_code=200, content={"status": "UP"})

@app.post("/setup")
async def setup_endpoint(request: Request):
    data = await request.json()
    return JSONResponse(status_code=200, content={"model_version": "yolo_backend_v1", "status": "Setup successful"})

@app.post("/webhook")
async def webhook(request: Request):
    """
    Handles webhooks from Label Studio:
    - Saves all responses in `data/json/`
    - Moves responses to the correct project subfolder based on project_id
    - Runs `convert_to_yolo()` to generate YOLO format labels
    - Triggers training when `START_TRAINING` is received
    """
    payload = await request.json()
    action = payload.get("action")
    print(f"📩 Webhook received for action: {action}")  # ✅ Debugging log
    logger = logging.getLogger(__name__)
    logger.info(f"Received webhook event: {action}")

    # Debugging log to check webhook calls
    print(f"Webhook received for action: {action}")

    # Save all responses in `data/json/`
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_filename = f"{action}_{timestamp}.json"
    json_filepath = os.path.join(JSON_DIR, json_filename)

    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    logger.info(f"Saved webhook response to {json_filepath}")

    # Handle project creation
    if action == "PROJECT_CREATED" or action == "PROJECT_UPDATED":
        project_data = payload.get("project", {})
        project_id = str(project_data.get("id"))
        project_name = project_data.get("title", "Unnamed Project")

        if project_id:
            create_project_folder(project_id)
            update_project_config(project_id, project_name)

    # Check if a project ID exists in the response
    project_data = payload.get("project", {})
    project_id = str(project_data.get("id")) if project_data else None

    if project_id:
        logger.info(f"Identified project ID: {project_id}")

        # Ensure the correct project folder exists
        project_folder = os.path.join(PROJECTS_DIR, project_id)
        os.makedirs(os.path.join(project_folder, "responses"), exist_ok=True)
        os.makedirs(os.path.join(project_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(project_folder, "labels"), exist_ok=True)

        # Move response JSON to project-specific folder
        project_response_file = os.path.join(project_folder, "responses", json_filename)
        os.rename(json_filepath, project_response_file)
        logger.info(f"Moved response to {project_response_file}")

        # Update `imageurl.txt` if an image is found
        image_path = payload.get("task", {}).get("data", {}).get("image")
        if image_path:
            update_image_url(image_path, project_folder)

        # If annotation created/updated, convert JSON to YOLO labels
        if action in ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"]:
            try:
                convert_to_yolo(project_response_file, project_folder)
                logger.info(f"Converted annotation to YOLO format for project {project_id}.")
            except Exception as e:
                logger.error(f"Error during conversion for project {project_id}: {e}")

    if action == "START_TRAINING":
        trigger_training()

    return JSONResponse(status_code=200, content={"status": "success", "message": f"Webhook {action} received and processed."})


def create_project_folder(project_id: str):
    """
    Ensures that the folder for a given project exists.
    Creates subdirectories for images, labels, and responses.
    """
    project_path = os.path.join(PROJECTS_DIR, project_id)
    os.makedirs(os.path.join(project_path, "responses"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "labels"), exist_ok=True)
    print(f"Created project folder: {project_path}")

def update_project_config(project_id: str, project_name: str):
    """
    Updates the projects config file when a new project is created.
    Default state is 'active': False.
    """
    # Ensure config file exists
    if not os.path.exists(CONFIG_FILE) or os.stat(CONFIG_FILE).st_size == 0:
        projects_config = {}  # Start with an empty dictionary if the file is missing or empty
    else:
        with open(CONFIG_FILE, "r") as f:
            try:
                projects_config = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Corrupt JSON detected in {CONFIG_FILE}. Resetting to empty.")
                projects_config = {}  # Reset if the file is corrupted

    # Update the dictionary with the new project
    projects_config[project_id] = {
        "name": project_name,
        "active": False,  # Initially set as inactive until user enables it for training
        "data_path": os.path.join(PROJECTS_DIR, project_id)
    }

    # Write the updated dictionary back to the file
    with open(CONFIG_FILE, "w") as f:
        json.dump(projects_config, f, indent=4)

    print(f"Updated project config: {CONFIG_FILE}")



def update_image_url(image_path: str, project_folder: str):
    """
    Updates the `imageurl.txt` file inside the corresponding project folder.
    - Ensures no duplicate entries.
    """
    txt_file_path = os.path.join(project_folder, "imageurl.txt")

    image_name = os.path.basename(image_path)
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

    print(f"Updated imageurl.txt for project {project_folder}")


def trigger_training():
    """
    Triggers training only for active projects once.
    """
    print("🔄 trigger_training() CALLED")  # ✅ Debugging log

    if not os.path.exists(CONFIG_FILE):
        print(f"Config file not found: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, "r") as f:
        projects_config = json.load(f)

    active_projects = {pid: p for pid, p in projects_config.items() if p["active"]}

    if not active_projects:
        print("No active projects available for training.")
        return

    print(f"🚀 Training triggered for {len(active_projects)} active projects.")

    # ✅ Run training only once per project
    yolo_model = YOLO()
    print("Projects Being Trained are : ",active_projects.items())
    for project_id, project_data in active_projects.items():
        project_path = project_data["data_path"]

        # if not os.path.exists(project_path):
        #     print(f"Skipping {project_data['name']}, project folder missing.")
        #     continue

        print(f"🔥 Starting training for project {project_id} ({project_data['name']})")  
        yolo_model.fit("START_TRAINING", {"project_id": project_id, "data_path": project_path})

    print("✅ Training completed for all selected projects.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
