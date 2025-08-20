# server/inference.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

class YOLOv5:
    """
    A class to run inference with a YOLOv5 ONNX model.
    """
    def __init__(self, model_path, confidence_thresh=0.5, iou_thresh=0.5):
        # Create an ONNX Runtime session to run the model
        self.session = ort.InferenceSession(model_path)
        
        # Get model input details
        self.model_inputs = self.session.get_inputs()
        self.input_shape = self.model_inputs[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]
        
        # Get model output details
        self.model_outputs = self.session.get_outputs()

        # Store thresholds
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh

        # COCO class names - this list must match the model's training data
        self.classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]

    def preprocess(self, image: Image.Image):
        """
        Preprocesses an image before running inference.
        Resizes, pads, and normalizes the image to match the model's input requirements.
        """
        img_width, img_height = image.size
        
        # Resize and pad the image to maintain its aspect ratio
        ratio = min(self.input_width / img_width, self.input_height / img_height)
        new_width, new_height = int(img_width * ratio), int(img_height * ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        padded_image = Image.new("RGB", (self.input_width, self.input_height), (114, 114, 114))
        padded_image.paste(resized_image, (0, 0))
        
        # Convert to a numpy array and normalize
        input_tensor = np.array(padded_image, dtype=np.float16) / 255.0
        
        # Change data layout from HWC to CHW
        input_tensor = input_tensor.transpose(2, 0, 1)
        
        # Add a batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, ratio, (0, 0) # No padding offset in this simple case

    def postprocess(self, output, ratio):
        """
        Postprocesses the model's output to get bounding boxes, scores, and class labels.
        This function is corrected to handle the specific output shape of YOLOv5.
        """
        # The output of the model is a single tensor of shape (batch_size, 25200, 85)
        # where 85 is [center_x, center_y, width, height, obj_conf, ...80 class_scores]
        predictions = np.squeeze(output[0])

        # Filter out detections with low object confidence.
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.confidence_thresh]
        obj_conf = obj_conf[obj_conf > self.confidence_thresh]

        if len(obj_conf) == 0:
            return []

        # Get the scores for all classes
        class_scores = predictions[:, 5:]
        # Get the class with the highest score for each detection
        class_ids = np.argmax(class_scores, axis=1)
        # Get the actual highest score
        max_scores = np.max(class_scores, axis=1)

        # Filter out detections where the max class score is below the confidence threshold
        keep = max_scores > self.confidence_thresh
        
        predictions = predictions[keep]
        class_ids = class_ids[keep]
        max_scores = max_scores[keep]

        if len(predictions) == 0:
            return []

        # Bounding boxes are in [center_x, center_y, width, height] format
        boxes = predictions[:, :4]

        # Rescale boxes from the model's input size (e.g., 640x640) back to the original image size
        boxes /= ratio

        # Convert from [center_x, center_y, width, height] to [x_min, y_min, x_max, y_max]
        x_min = boxes[:, 0] - boxes[:, 2] / 2
        y_min = boxes[:, 1] - boxes[:, 3] / 2
        x_max = boxes[:, 0] + boxes[:, 2] / 2
        y_max = boxes[:, 1] + boxes[:, 3] / 2
        
        # Combine the coordinates into a single array for NMS
        nms_boxes = np.column_stack((x_min, y_min, x_max, y_max))

        # Perform Non-Maximum Suppression
        indices = self.non_max_suppression(nms_boxes, max_scores, self.iou_thresh)

        # Format the final detections
        detections = []
        for i in indices:
            detections.append({
                "label": self.classes[class_ids[i]],
                "score": float(max_scores[i]),
                "xmin": float(nms_boxes[i, 0]),
                "ymin": float(nms_boxes[i, 1]),
                "xmax": float(nms_boxes[i, 2]),
                "ymax": float(nms_boxes[i, 3]),
            })

        return detections

    
    def detect(self, image_bytes: bytes):
        """
        The main detection method.
        """
        # Open the image from byte data
        image = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = image.size
        
        # Preprocess the image
        input_tensor, ratio, _ = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.model_inputs[0].name: input_tensor})
        
        # Postprocess the results
        detections = self.postprocess(outputs[0], ratio)

        # Normalize coordinates to [0, 1] as required by the project
        for det in detections:
            det["xmin"] /= original_width
            det["ymin"] /= original_height
            det["xmax"] /= original_width
            det["ymax"] /= original_height
            
        return detections

    @staticmethod
    def non_max_suppression(boxes, scores, iou_threshold):
        """
        A basic implementation of Non-Maximum Suppression.
        Expects boxes in [x_min, y_min, x_max, y_max] format.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep


# Create a single instance of the model to be used by the API
model = YOLOv5(model_path="yolov5n.onnx")