import os
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralyticsplus import YOLO



from ultralyticsplus import YOLO, render_result

from torchvision.ops import nms

from patchify_utils import patchify_image,display_image,PatchMatrix

"""
<View>
  <Image name='image' value='$image' zoom='true' zoomControl='true' rotateControl='true'/> 
  <RectangleLabels name='label' toName='image'>
    <Label value='esca' background='#FFA39E'/>
    <Label value='healthy' background='#5eeb00'/>
    <Label value='unknown' background='#ffffff'/>
    <Label value='pvl_healthy' background='#0335fc'/>
    <Label value='pvl_esca' background='#ffc800'>
    <Label value='pvl_unknown' background='#ee00ff'/>
    </RectangleLabels>
</View>
"""

MODEL_FILE_NAME = "best.pt"

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        """Configure any parameters of your model here
        """

        model = YOLO(os.path.join(os.environ.get("MODEL_DIR"), MODEL_FILE_NAME))

        # set model parameters
        model.overrides["conf"] = 0.25  # NMS confidence threshold
        model.overrides["iou"] = 0.45  # NMS IoU threshold
        model.overrides["agnostic_nms"] = False  # NMS class-agnostic
        model.overrides["max_det"] = 1000  # maximum number of detections per image

        self.model = model

        self.set("model_version", "0.0.2")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        """print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')"""

        predictions = []
        for task in tasks:
          prediction = self.predict_one_task(task)
          predictions.append(prediction)

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]

        return predictions

    # #simple prediction
    # def predict_one_task(self, task: Dict):
       
    #     path = self.get_local_path(task["data"]["image"], task_id=task["id"])
    #     print(f"\nTASK:\n{task}\n")
    #     model_results = self.model.predict(path)
    #     print(f"\nRESULT:\n{model_results}\n")
    #     results = []
    #     all_scores = []


    #     for i,row in enumerate(model_results[0].boxes.xyxyn):
    #         label=model_results[0].names[int(model_results[0].boxes.cls[i].item())]
    #         score = float(model_results[0].boxes.conf[i].item())
    #         results.append(
    #           {
    #               "from_name": "label",
    #               "source": "$image",
    #               "to_name": "image",
    #               "type": "rectanglelabels",
    #               "value": {
    #                   "height": (row[3].item() - row[1].item()) * 100,
    #                   "rectanglelabels": [label],
    #                   "rotation": 0,
    #                   "width": (row[2].item() - row[0].item()) * 100,
    #                   "x": row[0].item() * 100,
    #                   "y": row[1].item() * 100,
    #               },
    #               "score": score
    #           }
    #         )

    #         all_scores.append(score)

    #         i += 1

    #     avg_score = sum(all_scores) / max(len(all_scores), 1)

    #     return {"result": results, "score": avg_score, "model_version": self.get("model_version")}

    #pathicy prediction prediction
    def convert_bbox_format(self,x1, y1, x2, y2):
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        return width, height, x_center, y_center

    def predict_one_task(self, task: Dict):
        path = self.get_local_path(task["data"]["image"], task_id=task["id"])
        pm = PatchMatrix(self.model, path, (2000, 2000), (1000, 1000))
        print(f"\nTASK:\n{task}\n")
        model_results = {"model_version": self.get("model_result")}
        _, model_out = pm.merge_patches_and_boxes()  # self.model.predict(path)
        print(f"\nRESULT:\n{model_results}\n")
        results = []
        all_scores = []

        for i, row in enumerate(model_out):
            label = row[2]
            score = row[3]
            x1, y1, x2, y2 = tuple(row[0])
            width, height, x_center, y_center = self.convert_bbox_format(x1, y1, x2, y2)

            results.append(
                {
                    "from_name": "label",
                    "source": "$image",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": height,
                        "rectanglelabels": [label],
                        "rotation": 0,
                        "width": width,
                        "x": x_center,
                        "y": y_center,
                    },
                    "score": score,
                }
            )

            all_scores.append(score)

        avg_score = sum(all_scores) / max(len(all_scores), 1)

        return {"result": results, "score": avg_score, "model_version": self.get("model_version")}


    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
