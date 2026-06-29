import { getYoloScale, getYoloTask } from "./catalog";
import { formatBytes, formatGib, formatNumber } from "./format";
import { estimateParamsFromMetadata, getConfigNumber } from "./model";
import type {
  BreakdownItem,
  EstimateBand,
  FormulaRow,
  ModelKind,
  ModelMetadata,
  Precision,
  PrecisionConfig,
  RepositoryWorkload,
  RequirementReport,
  YoloConfig
} from "./types";

const GB = 1000 ** 3;
const MB = 1000 ** 2;

export function bytesPerPrecision(precision: Precision): number {
  if (precision === "fp32") return 4;
  if (precision === "bf16" || precision === "fp16") return 2;
  if (precision === "int8") return 1;
  return 0.5;
}

function band(typical: number, aggressive = 0.82, conservative = 1.32): EstimateBand {
  return {
    aggressive: typical * aggressive,
    typical,
    conservative: typical * conservative
  };
}

function metric(label: string, unit: string, typical: number, note?: string, aggressive?: number, conservative?: number) {
  return {
    label,
    unit,
    value: band(typical, aggressive, conservative),
    note
  };
}

function totalBytes(items: BreakdownItem[]): number {
  return items.reduce((sum, item) => sum + item.bytes, 0);
}

function gb(bytes: number): number {
  return bytes / GB;
}

function mb(bytes: number): number {
  return bytes / MB;
}

function safeParams(metadata: ModelMetadata | null, kind: ModelKind): number {
  return estimateParamsFromMetadata(metadata, kind) || 1_000_000_000;
}

function commonRepositoryWarnings(metadata: ModelMetadata | null, params: number | undefined): string[] {
  const warnings: string[] = [];
  if (!metadata) warnings.push("尚未加载模型仓库元数据，当前报告使用默认模型规模。");
  if (!metadata?.config) warnings.push("仓库未返回 config.json，结构参数使用默认值或由模型大小反推。");
  if (!params) warnings.push("无法直接获得参数量，已按模型文件大小和精度反推。");
  return warnings;
}

export function estimateRepositoryModel(
  kind: ModelKind,
  metadata: ModelMetadata | null,
  workload: RepositoryWorkload,
  precision: PrecisionConfig
): RequirementReport {
  if (kind === "asr") return estimateAsr(metadata, workload, precision);
  if (kind === "tts") return estimateTts(metadata, workload, precision);
  return estimateLlm(metadata, workload, precision);
}

function estimateLlm(
  metadata: ModelMetadata | null,
  workload: RepositoryWorkload,
  precision: PrecisionConfig
): RequirementReport {
  const config = metadata?.config || {};
  const params = safeParams(metadata, "llm");
  const explicitParams = estimateParamsFromMetadata(metadata, "llm");
  const weightBytes = params * bytesPerPrecision(precision.weights);
  const layers = getConfigNumber(config, ["num_hidden_layers", "n_layer", "num_layers"]) || 32;
  const hidden = getConfigNumber(config, ["hidden_size", "n_embd", "d_model"]) || 4096;
  const heads = getConfigNumber(config, ["num_attention_heads", "n_head"]) || 32;
  const kvHeads = getConfigNumber(config, ["num_key_value_heads", "num_kv_heads"]) || heads;
  const headDim = getConfigNumber(config, ["head_dim"]) || hidden / heads;
  const context = Math.max(workload.contextTokens, workload.promptTokens + workload.outputTokens);
  const kvBytes =
    workload.batchSize * context * layers * 2 * kvHeads * headDim * bytesPerPrecision(precision.kvCache);
  const activationBytes =
    workload.batchSize * Math.min(workload.promptTokens, 4096) * hidden * bytesPerPrecision(precision.activation) * 8;
  const runtimeBytes = Math.max(768 * MB, weightBytes * 0.06 + kvBytes * 0.08);
  const memoryItems: BreakdownItem[] = [
    { label: "模型权重", bytes: weightBytes, tone: "green" },
    { label: "KV cache", bytes: kvBytes, tone: "amber" },
    { label: "Prefill 激活/临时缓冲", bytes: activationBytes, tone: "blue" },
    { label: "运行时余量", bytes: runtimeBytes, tone: "gray" }
  ];
  const totalMemory = totalBytes(memoryItems);
  const effectiveWeightRead = weightBytes * Math.max(0.35, 1 / Math.sqrt(Math.max(1, workload.batchSize)));
  const decodeBandwidth = workload.targetTokensPerSecond * effectiveWeightRead;
  const computeTops = (2 * params * workload.targetTokensPerSecond) / 1e12;
  const storageBytes = Math.max(metadata?.storageBytes || 0, weightBytes) * 1.15;

  const formulas: FormulaRow[] = [
    {
      item: "权重容量",
      formula: "参数量 x 权重字节数",
      inputs: `${formatNumber(params / 1e9)}B params, ${precision.weights}`,
      result: formatGib(weightBytes)
    },
    {
      item: "KV cache",
      formula: "batch x context x layers x 2 x kv_heads x head_dim x KV字节数",
      inputs: `${workload.batchSize} x ${context} x ${layers} x 2 x ${kvHeads} x ${formatNumber(headDim)} x ${precision.kvCache}`,
      result: formatGib(kvBytes)
    },
    {
      item: "Decode 带宽",
      formula: "目标 tokens/s x 有效权重读取量",
      inputs: `${workload.targetTokensPerSecond} tokens/s, batch ${workload.batchSize}`,
      result: `${formatNumber(decodeBandwidth / GB)} GB/s`
    }
  ];

  return {
    title: `${metadata?.name || "LLM"} 推理硬件需求`,
    kind: "llm",
    confidence: explicitParams && metadata?.config ? "medium" : "low",
    summary: `典型配置需要约 ${formatNumber(gb(totalMemory))} GB 内存，decode 侧约 ${formatNumber(
      decodeBandwidth / GB
    )} GB/s 读带宽。`,
    metrics: [
      metric("总内存容量", "GB", gb(totalMemory), "含运行时安全余量"),
      metric("模型存储容量", "GB", gb(storageBytes), "模型文件、tokenizer 与部署文件"),
      metric("Decode 内存带宽", "GB/s", decodeBandwidth / GB, "低 batch 时通常是主要瓶颈"),
      metric("Decode 算力", "TOPS", computeTops, "按 2 x 参数量 x tokens/s 粗估"),
      metric("目标吞吐", "tokens/s", workload.targetTokensPerSecond, "聚合输出吞吐", 1, 1)
    ],
    memoryBreakdown: memoryItems,
    formulas,
    assumptions: [
      "推理框架会复用权重读取，batch 越大有效带宽压力越低。",
      "Prefill 临时内存按可控上界估算，不等同训练激活保存。",
      "保守值加入更高运行时和碎片余量，适合规格评审。"
    ],
    warnings: commonRepositoryWarnings(metadata, explicitParams)
  };
}

function estimateAsr(
  metadata: ModelMetadata | null,
  workload: RepositoryWorkload,
  precision: PrecisionConfig
): RequirementReport {
  const params = safeParams(metadata, "asr");
  const explicitParams = estimateParamsFromMetadata(metadata, "asr");
  const weightBytes = params * bytesPerPrecision(precision.weights);
  const frameBytes = workload.audioStreams * workload.sampleRate * 2 * Math.max(1, workload.targetRtf);
  const activationBytes = Math.max(256 * MB, weightBytes * 0.22);
  const runtimeBytes = Math.max(384 * MB, weightBytes * 0.18);
  const memoryItems: BreakdownItem[] = [
    { label: "模型权重", bytes: weightBytes, tone: "green" },
    { label: "Encoder/Decoder 激活", bytes: activationBytes, tone: "blue" },
    { label: "音频流缓存", bytes: frameBytes, tone: "amber" },
    { label: "运行时余量", bytes: runtimeBytes, tone: "gray" }
  ];
  const totalMemory = totalBytes(memoryItems);
  const realtimeFactor = workload.audioStreams / Math.max(0.1, workload.targetRtf);
  const computeTops = (params * 1.6 * realtimeFactor) / 1e12;
  const bandwidth = ((weightBytes * 0.18 + activationBytes * 4) * realtimeFactor) / GB;

  return {
    title: `${metadata?.name || "ASR"} 推理硬件需求`,
    kind: "asr",
    confidence: explicitParams ? "medium" : "low",
    summary: `典型配置需要约 ${formatNumber(gb(totalMemory))} GB 内存，目标实时率下约 ${formatNumber(
      computeTops
    )} TOPS。`,
    metrics: [
      metric("总内存容量", "GB", gb(totalMemory), "含音频缓存和运行时余量"),
      metric("模型存储容量", "GB", gb(Math.max(metadata?.storageBytes || 0, weightBytes) * 1.15), "公开仓库文件或权重反推"),
      metric("内存带宽", "GB/s", bandwidth, "流式 encoder/decoder 访问估算"),
      metric("算力", "TOPS", computeTops, "按目标 RTF 和并发流数估算"),
      metric("实时率目标", "RTF", workload.targetRtf, "越小代表越快", 1, 1)
    ],
    memoryBreakdown: memoryItems,
    formulas: [
      {
        item: "权重容量",
        formula: "参数量 x 权重字节数",
        inputs: `${formatNumber(params / 1e6)}M params, ${precision.weights}`,
        result: formatGib(weightBytes)
      },
      {
        item: "实时算力",
        formula: "参数量 x 经验计算系数 x 并发流数 / RTF",
        inputs: `${formatNumber(params / 1e6)}M, streams ${workload.audioStreams}, RTF ${workload.targetRtf}`,
        result: `${formatNumber(computeTops)} TOPS`
      }
    ],
    assumptions: [
      "ASR 估算按流式推理部署建模。",
      "Whisper/encoder-decoder 类模型会比纯 encoder 模型有更高解码开销。",
      "仓库缺少结构细节时使用模型大小反推参数量。"
    ],
    warnings: commonRepositoryWarnings(metadata, explicitParams)
  };
}

function estimateTts(
  metadata: ModelMetadata | null,
  workload: RepositoryWorkload,
  precision: PrecisionConfig
): RequirementReport {
  const params = safeParams(metadata, "tts");
  const explicitParams = estimateParamsFromMetadata(metadata, "tts");
  const weightBytes = params * bytesPerPrecision(precision.weights);
  const acousticBytes = weightBytes * 0.72;
  const vocoderBytes = weightBytes * 0.28;
  const activationBytes = Math.max(192 * MB, weightBytes * 0.2);
  const runtimeBytes = Math.max(256 * MB, weightBytes * 0.18);
  const memoryItems: BreakdownItem[] = [
    { label: "Acoustic model", bytes: acousticBytes, tone: "green" },
    { label: "Vocoder", bytes: vocoderBytes, tone: "amber" },
    { label: "生成缓存/激活", bytes: activationBytes, tone: "blue" },
    { label: "运行时余量", bytes: runtimeBytes, tone: "gray" }
  ];
  const totalMemory = totalBytes(memoryItems);
  const realtimeFactor = workload.audioStreams / Math.max(0.1, workload.targetRtf);
  const computeTops = (params * 2.2 * realtimeFactor) / 1e12;
  const bandwidth = ((weightBytes * 0.22 + activationBytes * 5) * realtimeFactor) / GB;

  return {
    title: `${metadata?.name || "TTS"} 推理硬件需求`,
    kind: "tts",
    confidence: explicitParams ? "medium" : "low",
    summary: `典型配置需要约 ${formatNumber(gb(totalMemory))} GB 内存，vocoder 侧实时生成约需 ${formatNumber(
      computeTops
    )} TOPS。`,
    metrics: [
      metric("总内存容量", "GB", gb(totalMemory), "acoustic model、vocoder 与缓存"),
      metric("模型存储容量", "GB", gb(Math.max(metadata?.storageBytes || 0, weightBytes) * 1.15), "部署包估算"),
      metric("内存带宽", "GB/s", bandwidth, "vocoder 通常造成持续访问压力"),
      metric("算力", "TOPS", computeTops, "按实时语音生成估算"),
      metric("实时率目标", "RTF", workload.targetRtf, "越小代表越快", 1, 1)
    ],
    memoryBreakdown: memoryItems,
    formulas: [
      {
        item: "权重容量",
        formula: "参数量 x 权重字节数",
        inputs: `${formatNumber(params / 1e6)}M params, ${precision.weights}`,
        result: formatGib(weightBytes)
      },
      {
        item: "生成算力",
        formula: "参数量 x vocoder经验系数 x 并发流数 / RTF",
        inputs: `${formatNumber(params / 1e6)}M, streams ${workload.audioStreams}, RTF ${workload.targetRtf}`,
        result: `${formatNumber(computeTops)} TOPS`
      }
    ],
    assumptions: [
      "TTS 按 acoustic model + vocoder 两段部署估算。",
      "未能识别模型结构时，按公开模型文件大小反推参数量。",
      "多说话人、长文本和高采样率会增加运行缓存。"
    ],
    warnings: commonRepositoryWarnings(metadata, explicitParams)
  };
}

export function estimateYolo(config: YoloConfig): RequirementReport {
  const scale = getYoloScale(config.modelId);
  const task = getYoloTask(config.task);
  const precisionBytes = bytesPerPrecision(config.precision);
  const classFactor = config.task === "classify" ? config.classes / 1000 : config.classes / 80;
  const headAdjustment = 1 + Math.max(-0.12, Math.min(0.35, (classFactor - 1) * 0.08));
  const params = scale.paramsM * 1e6 * task.paramsMultiplier * headAdjustment;
  const weightBytes = params * precisionBytes;
  const sizeFactor = (config.imageSize / 640) ** 2;
  const gflopsPerFrame = scale.gflops640 * sizeFactor * task.flopsMultiplier * headAdjustment;
  const activationBytes =
    config.batchSize * config.imageSize * config.imageSize * precisionBytes * task.activationFactor * task.flopsMultiplier;
  const candidateCount =
    config.task === "classify"
      ? config.classes
      : Math.round((config.imageSize / 8) ** 2 + (config.imageSize / 16) ** 2 + (config.imageSize / 32) ** 2);
  const outputValues = config.task === "classify" ? config.classes : config.classes + task.outputExtraValues;
  const outputBytes = config.batchSize * candidateCount * outputValues * precisionBytes;
  const postprocessBytes =
    config.task === "classify" ? config.classes * 4 : config.batchSize * candidateCount * outputValues * 4 * 1.8;
  const runtimeBytes = Math.max(256 * MB, weightBytes * 0.18 + activationBytes * 0.45 + outputBytes);
  const memoryItems: BreakdownItem[] = [
    { label: "模型权重", bytes: weightBytes, tone: "green" },
    { label: "特征图峰值", bytes: activationBytes, tone: "blue" },
    { label: "输出 tensor", bytes: outputBytes, tone: "amber" },
    { label: "后处理/NMS 工作区", bytes: postprocessBytes, tone: "red" },
    { label: "运行时余量", bytes: runtimeBytes, tone: "gray" }
  ];
  const totalMemory = totalBytes(memoryItems);
  const rawTops = (gflopsPerFrame * config.targetFps * config.batchSize) / 1000;
  const backendEfficiency = {
    generic: 0.35,
    onnxruntime: 0.45,
    tensorrt: 0.62,
    npu: 0.52
  }[config.backend];
  const requiredTops = rawTops / backendEfficiency;
  const bandwidthBytes =
    config.targetFps *
    config.batchSize *
    (weightBytes * 0.18 + activationBytes * 3.2 + outputBytes * 2.4 + postprocessBytes * 1.1);
  const storageBytes = weightBytes * 1.25;

  return {
    title: `${scale.label} ${task.label} 推理硬件需求`,
    kind: "yolo",
    confidence: "medium",
    summary: `在 ${config.imageSize}x${config.imageSize}、${config.targetFps} FPS 下，典型需要约 ${formatNumber(
      gb(totalMemory)
    )} GB 运行内存和 ${formatNumber(requiredTops)} TOPS。`,
    metrics: [
      metric("总内存容量", "GB", gb(totalMemory), "含特征图、输出和后处理工作区"),
      metric("模型存储容量", "GB", gb(storageBytes), "权重与部署文件"),
      metric("主网络算力", "TOPS", requiredTops, `按 ${config.backend} 有效利用率估算`),
      metric("DDR/显存带宽", "GB/s", bandwidthBytes / GB, "含特征图和后处理访问"),
      metric("目标帧率", "FPS", config.targetFps, "单路或聚合帧率", 1, 1),
      metric("后处理工作区", "MB", mb(postprocessBytes), task.postprocessLabel)
    ],
    memoryBreakdown: memoryItems,
    formulas: [
      {
        item: "YOLO FLOPs",
        formula: "640基准GFLOPs x (输入尺寸/640)^2 x 任务系数",
        inputs: `${scale.gflops640} GFLOPs, ${config.imageSize}px, ${task.label}`,
        result: `${formatNumber(gflopsPerFrame)} GFLOPs/frame`
      },
      {
        item: "实时算力",
        formula: "GFLOPs/frame x FPS x batch / 后端有效利用率",
        inputs: `${formatNumber(gflopsPerFrame)} x ${config.targetFps} x ${config.batchSize} / ${formatNumber(
          backendEfficiency
        )}`,
        result: `${formatNumber(requiredTops)} TOPS`
      },
      {
        item: "候选框/输出",
        formula: "多尺度候选数 x 输出字段 x 精度字节数",
        inputs: `${candidateCount} candidates, ${outputValues} values, ${config.precision}`,
        result: formatBytes(outputBytes)
      }
    ],
    assumptions: [
      "YOLO 目录使用 Ultralytics 当前主线模型规模，任务 head 通过系数调整。",
      "后处理单独估算，Detection/Segmentation/Pose/OBB 与 Classification 的工作区不同。",
      "自定义类别数主要影响 head、输出 tensor 和后处理，不重估完整 backbone。"
    ],
    warnings: [
      "第一版不解析 .pt、YAML 或 ONNX，实际自定义模型请用实测 FLOPs 校准。",
      "NMS 在 CPU 或低端 NPU 上可能成为延迟瓶颈，评审时应单独验证。"
    ]
  };
}
