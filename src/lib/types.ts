export type ModelKind = "llm" | "asr" | "tts" | "yolo";
export type ModelSource = "huggingface" | "modelscope";
export type Precision = "fp32" | "bf16" | "fp16" | "int8" | "int4";
export type YoloTask = "detect" | "segment" | "pose" | "obb" | "classify";
export type EstimateKey = "aggressive" | "typical" | "conservative";

export interface ModelFile {
  name: string;
  size?: number;
  lfs?: {
    size?: number;
  };
}

export interface ModelMetadata {
  source: ModelSource;
  modelId: string;
  name: string;
  pipelineTag?: string;
  tags: string[];
  config?: Record<string, unknown> | null;
  files: ModelFile[];
  storageBytes?: number;
  raw?: Record<string, unknown>;
}

export interface PrecisionConfig {
  weights: Precision;
  activation: Precision;
  kvCache: Precision;
}

export interface RepositoryWorkload {
  batchSize: number;
  benchmarkTokens: number;
  prefillTokensPerSecond: number;
  promptTokens: number;
  targetPrefillMs: number;
  contextTokens: number;
  outputTokens: number;
  targetTokensPerSecond: number;
  audioStreams: number;
  targetRtf: number;
  sampleRate: number;
  imageSize: number;
  targetFps: number;
}

export interface YoloConfig {
  modelId: string;
  task: YoloTask;
  imageSize: number;
  batchSize: number;
  targetFps: number;
  classes: number;
  precision: Precision;
  backend: "generic" | "onnxruntime" | "tensorrt" | "npu";
}

export interface EstimateBand {
  aggressive: number;
  typical: number;
  conservative: number;
}

export interface ReportMetric {
  label: string;
  unit: string;
  value: EstimateBand;
  note?: string;
}

export interface BreakdownItem {
  label: string;
  bytes: number;
  tone: "green" | "amber" | "blue" | "red" | "gray";
}

export interface FormulaRow {
  item: string;
  formula: string;
  inputs: string;
  result: string;
  level?: "key" | "advanced";
}

export interface RequirementReport {
  title: string;
  kind: ModelKind;
  confidence: "high" | "medium" | "low";
  summary: string;
  metrics: ReportMetric[];
  memoryBreakdown: BreakdownItem[];
  formulas: FormulaRow[];
  assumptions: string[];
  warnings: string[];
}
