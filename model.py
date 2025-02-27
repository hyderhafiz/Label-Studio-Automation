import os
import json
import logging
from pathlib import Path
from label_studio_ml.model import LabelStudioMLBase
# from label_studio_ml.response import ModelResponse
import time

# from control_models.base import ControlModel
# from control_models.choices import ChoicesModel
# from control_models.rectangle_labels import RectangleLabelsModel
# from control_models.rectangle_labels_obb import RectangleLabelsObbModel
# from control_models.polygon_labels import PolygonLabelsModel
# from control_models.keypoint_labels import KeypointLabelsModel
# from control_models.video_rectangle import VideoRectangleModel
# from control_models.timeline_labels import TimelineLabelsModel
# from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

# Register available control model classes
# available_model_classes = [
#     ChoicesModel,
#     RectangleLabelsModel,
#     RectangleLabelsObbModel,
#     PolygonLabelsModel,
#     KeypointLabelsModel,
#     VideoRectangleModel,
#     TimelineLabelsModel,
# ]


def send_slack_notification(current_evaluation: dict, prev_evaluation: dict, comparison_message: str):
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        try:
            import requests
            # Build the Slack message.
            slack_message = "YOLO Training Evaluation Comparison:\n"
            # Get previous and current mAP50 scores:
            prev_mAP50 = prev_evaluation.get("metrics/mAP50(B)") if prev_evaluation else None
            current_mAP50 = current_evaluation.get("metrics/mAP50(B)") if current_evaluation else None

            if prev_mAP50 is not None:
                slack_message += f"*Previous mAP50:* {prev_mAP50}\n"
            else:
                slack_message += "*Previous mAP50:* Not available\n"
            if current_mAP50 is not None:
                slack_message += f"*Current mAP50:* {current_mAP50}\n"
            else:
                slack_message += "*Current mAP50:* Not available\n"

            slack_message += "\n*Comparison:* " + comparison_message + "\n"
            slack_message += "\n*Current Evaluation Metrics:*\n```" + json.dumps(current_evaluation, indent=2) + "```"
            if prev_evaluation:
                slack_message += "\n*Previous Evaluation Metrics:*\n```" + json.dumps(prev_evaluation, indent=2) + "```"
            payload = {"text": slack_message}
            response = requests.post(slack_webhook_url, json=payload)
            if response.status_code == 200:
                print("Sent evaluation metrics to Slack successfully.")
            else:   
                print("Failed to send Slack message. Status code:", response.status_code, "Response:", response.text)
        except Exception as e:
            print("Error sending Slack message:", e)
    else:
        print("SLACK_WEBHOOK_URL not set; skipping Slack notification.")

class YOLO(LabelStudioMLBase):
    """Label Studio ML Backend based on Ultralytics YOLO for vehicle detection."""

    def setup(self):
        self.set("model_version", "YOLOv8Detector-v0.0.1")

    # def detect_control_models(self) -> List[ControlModel]:
    #     control_models = []
    #     for control in self.label_interface.controls:
    #         if not control.to_name:
    #             logger.warning(f'{control.tag} {control.name} has no "toName" attribute, skipping it')
    #             continue
    #         for model_class in available_model_classes:
    #             if model_class.is_control_matched(control):
    #                 instance = model_class.create(self, control)
    #                 if not instance:
    #                     logger.debug(f"No instance created for {control.tag} {control.name}")
    #                     continue
    #                 if not instance.label_map:
    #                     logger.error(f"No label map built for the '{control.tag}' control tag '{instance.from_name}'.")
    #                     continue
    #                 control_models.append(instance)
    #                 logger.debug(f"Control tag with model detected: {instance}")
    #                 break
    #     if not control_models:
    #         control_tags = ", ".join([c.type for c in available_model_classes])
    #         raise ValueError(f"No suitable control tags (e.g. {control_tags}) detected in the label config:\n{self.label_config}")
    #     return control_models

    # def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
    #     logger.info(f"Run prediction on {len(tasks)} tasks, project ID = {self.project_id}")
    #     control_models = self.detect_control_models()
    #     predictions = []
    #     for task in tasks:
    #         regions = []
    #         for model in control_models:
    #             path = model.get_path(task)
    #             regions += model.predict_regions(path)
    #         all_scores = [region["score"] for region in regions if "score" in region]
    #         avg_score = sum(all_scores) / max(len(all_scores), 1)
    #         prediction = {
    #             "result": regions,
    #             "score": avg_score,
    #             "model_version": self.model_version,
    #         }
    #         predictions.append(prediction)
    #     return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        Trigger YOLO training, evaluate on the train/val set, and then save and compare evaluation metrics.
        Finally, send both the current and previous evaluations along with the comparison to Slack.
        """
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            print(f"Skipping training for event {event}")
            return {"message": f"Event {event} is not supported for training."}

        # (Assume label_config is already set, or set a default if needed.)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print("Training directory:", base_dir)

        images_dir = os.path.join(base_dir, "data/images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images folder not found at {images_dir}")

        # Create YAML file for YOLO training.
        names_list = ['Ambulance', 'Car', 'Bus', 'Truck', 'Motorcycle']
        nc = len(names_list)
        data_yaml = os.path.join(base_dir, "yolo_data.yaml")
        yaml_content = f"train: {images_dir}\nval: {images_dir}\nnc: {nc}\nnames: {names_list}\n"
        with open(data_yaml, "w") as f:
            f.write(yaml_content)
        print("Created YOLO data YAML file at", data_yaml)

        # Load YOLO model from checkpoint if available; otherwise load base model.
        from ultralytics import YOLO as YOLOv8
        checkpoint_path = os.path.join(base_dir, "yolov8_vehicle.pt")
        if os.path.exists(checkpoint_path):
            print("Loading fine-tuned model from checkpoint:", checkpoint_path)
            model = YOLOv8(checkpoint_path)
        else:
            print("Loading base YOLOv8 model.")
            model = YOLOv8("yolov8n-seg.pt")

        print("Starting YOLOv8 training...")
        model.train(
            data=data_yaml,
            epochs=50,      # Adjust epochs as needed
            imgsz=640,     # Adjust image size if needed
            project=base_dir,
            name="yolov8_vehicle",
            exist_ok=True
        )

        final_ckpt = os.path.join(base_dir, "yolov8_vehicle.pt")
        model.save(final_ckpt)
        print("Training complete. Model saved to", final_ckpt)

        # Evaluate the trained model on the training set.
        print("Evaluating model on the training set...")
        try:
            eval_results = model.val(data=data_yaml, imgsz=640)
            # Use the results_dict attribute (a dictionary of metrics).
            evaluation = eval_results.results_dict
            print("Evaluation results:", evaluation)
        except Exception as e:
            print("Error during evaluation:", e)
            evaluation = {"error": str(e)}

        # Save evaluation metrics.
        current_eval_dir = os.path.join(base_dir, "evaluation/eval_current")
        previous_eval_dir = os.path.join(base_dir, "evaluation/eval_previous")
        os.makedirs(current_eval_dir, exist_ok=True)
        os.makedirs(previous_eval_dir, exist_ok=True)

        current_eval_file = os.path.join(current_eval_dir, "evaluation.json")
        prev_evaluation = None  # Default to None if no previous evaluation.
        if os.path.exists(current_eval_file):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            previous_eval_file = os.path.join(previous_eval_dir, f"evaluation_{timestamp}.json")
            os.rename(current_eval_file, previous_eval_file)
            print("Moved previous evaluation to", previous_eval_file)
            # Load the previous evaluation.
            with open(previous_eval_file, "r") as f:
                prev_evaluation = json.load(f)

        with open(current_eval_file, "w") as f:
            json.dump(evaluation, f, indent=4)
        print("Saved current evaluation metrics to", current_eval_file)

        # Compare mAP50 scores.
        prev_mAP50 = None
        current_mAP50 = evaluation.get("metrics/mAP50(B)")
        if prev_evaluation:
            prev_mAP50 = prev_evaluation.get("metrics/mAP50(B)")
        if prev_mAP50 is not None and current_mAP50 is not None:
            if current_mAP50 > prev_mAP50:
                comparison_message = f"Improved: mAP50 increased from {prev_mAP50} to {current_mAP50}"
            else:
                comparison_message = f"Not improved: mAP50 decreased or remained the same from {prev_mAP50} to {current_mAP50}"
            print(comparison_message)
        else:
            comparison_message = "Could not compare mAP50 scores; one of them is missing."
            print(comparison_message)

        # Send evaluation metrics and comparison to Slack.
        send_slack_notification(evaluation, prev_evaluation, comparison_message)

        return {"message": "Training complete.", "checkpoint": final_ckpt, "evaluation": evaluation}

