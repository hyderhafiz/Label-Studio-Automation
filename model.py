import os
import json
import logging
import time
import yaml
import requests
from pathlib import Path
from ultralytics import YOLO as YOLOv8

logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/projects_config.json")

def send_slack_notification(current_evaluation: dict, prev_evaluation: dict, comparison_message: str):
    """
    Sends a Slack notification with evaluation metrics.
    """
    slack_webhook_url = "https://hooks.slack.com/services/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"
    if slack_webhook_url:
        try:
            slack_message = "YOLO Training Evaluation Comparison:\n"
            
            prev_mAP50 = prev_evaluation.get("metrics/mAP50(B)") if prev_evaluation else None
            current_mAP50 = current_evaluation.get("metrics/mAP50(B)") if current_evaluation else None

            slack_message += f"*Previous mAP50:* {prev_mAP50 if prev_mAP50 is not None else 'Not available'}\n"
            slack_message += f"*Current mAP50:* {current_mAP50 if current_mAP50 is not None else 'Not available'}\n"
            slack_message += f"\n*Comparison:* {comparison_message}\n"

            slack_message += "\n*Current Evaluation Metrics:*\n```" + json.dumps(current_evaluation, indent=2) + "```"
            if prev_evaluation:
                slack_message += "\n*Previous Evaluation Metrics:*\n```" + json.dumps(prev_evaluation, indent=2) + "```"

            response = requests.post(slack_webhook_url, json={"text": slack_message})
            if response.status_code == 200:
                print("Sent evaluation metrics to Slack successfully.")
            else:
                print(f"Failed to send Slack message. Status code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print("Error sending Slack message:", e)
    else:
        print("SLACK_WEBHOOK_URL not set; skipping Slack notification.")

class YOLO:
    """YOLO training handler that only runs for selected projects."""

    def setup(self):
        self.set("model_version", "YOLOv8Detector-v0.0.1")

    def load_active_projects(self):
        """
        Load project configuration and filter only active projects.
        """
        if not os.path.exists(CONFIG_FILE):
            print(f"Config file not found: {CONFIG_FILE}")
            return {}

        with open(CONFIG_FILE, "r") as f:
            projects_config = json.load(f)

        return {pid: p for pid, p in projects_config.items() if p["active"]}

    def fit(self, event, data, **kwargs):
        """
        Run training only for active projects, evaluate results, and send Slack notifications.
        """
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            print(f"Skipping training for event {event}")
            return {"message": f"Event {event} is not supported for training."}

        # Prevent duplicate training runs
        if hasattr(self, "training_started") and self.training_started:
            print("⚠️ Training already running. Skipping duplicate request.")
            return {"message": "Training already running, skipping duplicate execution."}

        self.training_started = True  # Prevent re-trigger

        # Load active projects
        active_projects = self.load_active_projects()
        if not active_projects:
            print("No active projects available for training.")
            return {"message": "No active projects available for training."}

        trained_projects = set()  # Track projects that have been trained

        for project_id, project_data in active_projects.items():

            trained_projects.add(project_id)  #  Mark project as trained

            project_path = project_data["data_path"]

            # Ensure the path is a directory (not a JSON file)
            if project_path.endswith(".json"):
                project_path = os.path.dirname(project_path)

            if not os.path.exists(project_path):
                print(f"Skipping {project_data['name']}, project folder missing.")
                continue

            images_dir = os.path.join(project_path, "images")
            labels_dir = os.path.join(project_path, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                print(f"Skipping {project_data['name']}, missing data folders.")
                continue

            # Create YOLO data YAML file
            data_yaml = os.path.join(project_path, "yolo_data.yaml")
            yaml_dict = {
                "train": images_dir.replace("\\", "/"),
                "val": images_dir.replace("\\", "/"),
                "nc": 5,
                "names": ['Ambulance', 'Car', 'Bus', 'Truck', 'Motorcycle']
            }

            with open(data_yaml, "w") as f:
                yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

            print(f"Created YOLO data YAML file at {data_yaml}")

            # Load YOLO Model
            checkpoint_path = os.path.join(project_path, "yolov8_vehicle.pt")
            if os.path.exists(checkpoint_path):
                print(f"Loading fine-tuned model from checkpoint: {checkpoint_path}")
                model = YOLOv8(checkpoint_path)
            else:
                print("Loading base YOLOv8 model.")
                model = YOLOv8("yolov8n.pt")

            print(f"🚀 Training model for Project {project_id}...")
            model.train(
                data=data_yaml,
                epochs=3,  # Adjust as needed
                imgsz=640,
                project=project_path,
                name="yolov8_vehicle",
                exist_ok=True
            )

            final_ckpt = os.path.join(project_path, "yolov8_vehicle.pt")
            model.save(final_ckpt)
            print(f"Training complete for {project_data['name']}. Model saved to {final_ckpt}")

            # Evaluate Model Performance
            print("Evaluating model on the training set...")
            try:
                eval_results = model.val(data=data_yaml, imgsz=640)
                evaluation = eval_results.results_dict
                print("Evaluation results:", evaluation)
            except Exception as e:
                print("Error during evaluation:", e)
                evaluation = {"error": str(e)}

            # Save Evaluation Metrics
            current_eval_dir = os.path.join(project_path, "evaluation/eval_current")
            previous_eval_dir = os.path.join(project_path, "evaluation/eval_previous")
            os.makedirs(current_eval_dir, exist_ok=True)
            os.makedirs(previous_eval_dir, exist_ok=True)

            current_eval_file = os.path.join(current_eval_dir, "evaluation.json")
            prev_evaluation = None

            if os.path.exists(current_eval_file):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                previous_eval_file = os.path.join(previous_eval_dir, f"evaluation_{timestamp}.json")
                os.rename(current_eval_file, previous_eval_file)
                print(f"Moved previous evaluation to {previous_eval_file}")

                with open(previous_eval_file, "r") as f:
                    prev_evaluation = json.load(f)

            with open(current_eval_file, "w") as f:
                json.dump(evaluation, f, indent=4)
            print(f"Saved current evaluation metrics to {current_eval_file}")

            # Compare mAP50 Scores
            prev_mAP50 = prev_evaluation.get("metrics/mAP50(B)") if prev_evaluation else None
            current_mAP50 = evaluation.get("metrics/mAP50(B)")

            if prev_mAP50 is not None and current_mAP50 is not None:
                if current_mAP50 > prev_mAP50:
                    comparison_message = f"Improved: mAP50 increased from {prev_mAP50} to {current_mAP50}"
                else:
                    comparison_message = f"Not improved: mAP50 decreased or remained the same from {prev_mAP50} to {current_mAP50}"
                print(comparison_message)
            else:
                comparison_message = "Could not compare mAP50 scores; one of them is missing."
                print(comparison_message)

            # Send Evaluation Metrics and Comparison to Slack
            send_slack_notification(evaluation, prev_evaluation, comparison_message)

        return {"message": "Training completed for selected projects."}

