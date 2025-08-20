// Utility to convert Float32Array to Float16 (Uint16Array)
function float32ToFloat16Buffer(float32Array: Float32Array): Uint16Array {
  // Adapted from https://stackoverflow.com/a/56728142
  const uint16Array = new Uint16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    let f = float32Array[i];
    let sign = (f < 0) ? 1 : 0;
    f = Math.abs(f);
    if (isNaN(f)) {
      uint16Array[i] = 0x7e00;
      continue;
    }
    if (f === Infinity) {
      uint16Array[i] = (sign << 15) | 0x7c00;
      continue;
    }
    if (f === 0) {
      uint16Array[i] = (sign << 15);
      continue;
    }
    let exp = Math.floor(Math.log2(f));
    let frac = f / Math.pow(2, exp) - 1;
    let halfExp = exp + 15;
    if (halfExp <= 0) {
      // subnormal
      uint16Array[i] = (sign << 15) | Math.round(frac * Math.pow(2, 10 + exp + 14));
    } else if (halfExp >= 31) {
      // overflow
      uint16Array[i] = (sign << 15) | 0x7c00;
    } else {
      uint16Array[i] = (sign << 15) | (halfExp << 10) | Math.round(frac * 1024);
    }
  }
  return uint16Array;
}
// frontend/src/lib/inference.ts
import * as ort from "onnxruntime-web";

// Define the structure of our detection data, same as on the server
export interface Detection {
  label: string;
  score: number;
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

// COCO class names, same as on the server
const CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
];

export class YOLOv5_WASM {
  private session: ort.InferenceSession | null = null;
  private modelInputShape = [1, 3, 640, 640]; // Default for YOLOv5

  // Initialize the ONNX Runtime session
  async init(modelPath: string): Promise<void> {
    // It's important to set the wasm paths for onnxruntime-web
    ort.env.wasm.wasmPaths = "/static/wasm/";
    
    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["wasm"],
    });
    console.log("WASM session initialized.");
  }

  // Preprocess the image data from a canvas
  private preprocess(ctx: CanvasRenderingContext2D): [ort.Tensor, number] {
    const { width: canvasWidth, height: canvasHeight } = ctx.canvas;
    const modelWidth = this.modelInputShape[3];
    const modelHeight = this.modelInputShape[2];

    const ratio = Math.min(modelWidth / canvasWidth, modelHeight / canvasHeight);
    const newWidth = Math.round(canvasWidth * ratio);
    const newHeight = Math.round(canvasHeight * ratio);

    // Create a temporary canvas for resizing
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = newWidth;
    tempCanvas.height = newHeight;
    const tempCtx = tempCanvas.getContext("2d")!;
    tempCtx.drawImage(ctx.canvas, 0, 0, newWidth, newHeight);
    const imageData = tempCtx.getImageData(0, 0, newWidth, newHeight);

    const red: number[] = [], green: number[] = [], blue: number[] = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
      red.push(imageData.data[i] / 255);
      green.push(imageData.data[i + 1] / 255);
      blue.push(imageData.data[i + 2] / 255);
    }

    const transposedData = red.concat(green, blue);
    const float32Data = new Float32Array(this.modelInputShape[1] * modelWidth * modelHeight);
    
    // Fill the tensor with the transposed image data and padding
    float32Data.fill(114 / 255); // Fill with padding color
    for (let i = 0; i < transposedData.length; i++) {
        const ch = Math.floor(i / (newWidth * newHeight));
        const y = Math.floor((i % (newWidth * newHeight)) / newWidth);
        const x = i % newWidth;
        float32Data[ch * (modelWidth * modelHeight) + y * modelWidth + x] = transposedData[i];
    }

  // Convert to float16 to match model input (as in server)
  const float16Data = float32ToFloat16Buffer(float32Data);
  const inputTensor = new ort.Tensor("float16", float16Data, this.modelInputShape);
  return [inputTensor, ratio];
  }

  // Postprocess the model output
  private postprocess(output: ort.Tensor, ratio: number): Detection[] {
    const predictions = output.data as Float32Array;
    const detections: Detection[] = [];

    // The output is (1, 25200, 85)
    // We iterate through all 25200 possible detections
    for (let i = 0; i < 25200; i++) {
      const objConfidence = predictions[i * 85 + 4];
      if (objConfidence < 0.5) continue; // Confidence threshold

      const classScores = predictions.slice(i * 85 + 5, (i + 1) * 85);
      let maxScore = 0;
      let classId = -1;

      for (let j = 0; j < classScores.length; j++) {
        if (classScores[j] > maxScore) {
          maxScore = classScores[j];
          classId = j;
        }
      }

      if (maxScore > 0.5) { // Class score threshold
        const [x, y, w, h] = predictions.slice(i * 85, i * 85 + 4);
        
        const x_min = (x - w / 2) / ratio;
        const y_min = (y - h / 2) / ratio;
        const x_max = (x + w / 2) / ratio;
        const y_max = (y + h / 2) / ratio;
        
        detections.push({
          label: CLASSES[classId],
          score: maxScore,
          xmin: x_min,
          ymin: y_min,
          xmax: x_max,
          ymax: y_max,
        });
      }
    }
    // Note: A proper implementation would add Non-Maximum Suppression here.
    // For simplicity, we are skipping it, which might result in overlapping boxes.
    return detections;
  }

  // The main detection function
  async detect(ctx: CanvasRenderingContext2D): Promise<Detection[]> {
    if (!this.session || !ctx) return [];
    
    const originalWidth = ctx.canvas.width;
    const originalHeight = ctx.canvas.height;
    
    const [inputTensor, ratio] = this.preprocess(ctx);
    const feeds = { [this.session.inputNames[0]]: inputTensor };
    const results = await this.session.run(feeds);
    
    const detections = this.postprocess(results[this.session.outputNames[0]], ratio);

    // Normalize coordinates to [0, 1]
    return detections.map(det => ({
      ...det,
      xmin: det.xmin / originalWidth,
      ymin: det.ymin / originalHeight,
      xmax: det.xmax / originalWidth,
      ymax: det.ymax / originalHeight,
    }));
  }
}