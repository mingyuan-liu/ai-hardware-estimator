import type { YoloTask } from "./types";

export interface YoloScale {
  id: string;
  label: string;
  paramsM: number;
  gflops640: number;
}

export interface YoloTaskProfile {
  id: YoloTask;
  label: string;
  paramsMultiplier: number;
  flopsMultiplier: number;
  activationFactor: number;
  outputExtraValues: number;
  postprocessLabel: string;
}

export const yoloScales: YoloScale[] = [
  { id: "yolov8n", label: "YOLOv8n", paramsM: 3.2, gflops640: 8.7 },
  { id: "yolov8s", label: "YOLOv8s", paramsM: 11.2, gflops640: 28.6 },
  { id: "yolov8m", label: "YOLOv8m", paramsM: 25.9, gflops640: 78.9 },
  { id: "yolov8l", label: "YOLOv8l", paramsM: 43.7, gflops640: 165.2 },
  { id: "yolov8x", label: "YOLOv8x", paramsM: 68.2, gflops640: 257.8 },
  { id: "yolo11n", label: "YOLO11n", paramsM: 2.6, gflops640: 6.5 },
  { id: "yolo11s", label: "YOLO11s", paramsM: 9.4, gflops640: 21.5 },
  { id: "yolo11m", label: "YOLO11m", paramsM: 20.1, gflops640: 68.0 },
  { id: "yolo11l", label: "YOLO11l", paramsM: 25.3, gflops640: 86.9 },
  { id: "yolo11x", label: "YOLO11x", paramsM: 56.9, gflops640: 194.9 }
];

export const yoloTasks: YoloTaskProfile[] = [
  {
    id: "detect",
    label: "Detection",
    paramsMultiplier: 1,
    flopsMultiplier: 1,
    activationFactor: 18,
    outputExtraValues: 5,
    postprocessLabel: "decode + NMS"
  },
  {
    id: "segment",
    label: "Segmentation",
    paramsMultiplier: 1.18,
    flopsMultiplier: 1.28,
    activationFactor: 30,
    outputExtraValues: 37,
    postprocessLabel: "decode + NMS + mask compose"
  },
  {
    id: "pose",
    label: "Pose",
    paramsMultiplier: 1.1,
    flopsMultiplier: 1.14,
    activationFactor: 23,
    outputExtraValues: 56,
    postprocessLabel: "decode + NMS + keypoint decode"
  },
  {
    id: "obb",
    label: "OBB",
    paramsMultiplier: 1.08,
    flopsMultiplier: 1.1,
    activationFactor: 21,
    outputExtraValues: 6,
    postprocessLabel: "decode + rotated NMS"
  },
  {
    id: "classify",
    label: "Classification",
    paramsMultiplier: 0.86,
    flopsMultiplier: 0.74,
    activationFactor: 12,
    outputExtraValues: 0,
    postprocessLabel: "top-k"
  }
];

export function getYoloScale(id: string): YoloScale {
  return yoloScales.find((scale) => scale.id === id) || yoloScales[0];
}

export function getYoloTask(id: YoloTask): YoloTaskProfile {
  return yoloTasks.find((task) => task.id === id) || yoloTasks[0];
}
