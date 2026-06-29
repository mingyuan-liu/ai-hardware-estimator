import { getYoloScale, getYoloTask } from "./catalog";
import { formatBytes, formatComputeOps, formatComputeTops, formatGib, formatNumber } from "./format";
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
  const prefillAttentionOps =
    4 * workload.batchSize * layers * workload.promptTokens * workload.promptTokens * hidden;
  const prefillOps = 2 * params * workload.batchSize * workload.promptTokens + prefillAttentionOps;
  const targetPrefillSeconds = Math.max(0.001, workload.targetPrefillMs / 1000);
  const prefillComputeTops = prefillOps / targetPrefillSeconds / 1e12;
  const decodeSingleTokensPerSecond = workload.targetTokensPerSecond;
  const decodeAggregateTokensPerSecond = decodeSingleTokensPerSecond * workload.batchSize;
  const effectiveWeightRead = weightBytes * Math.max(0.35, 1 / Math.sqrt(Math.max(1, workload.batchSize)));
  const decodeWeightBandwidth = decodeAggregateTokensPerSecond * effectiveWeightRead;
  const kvBytesPerToken =
    workload.batchSize * layers * 2 * kvHeads * headDim * bytesPerPrecision(precision.kvCache);
  const decodeKvBandwidth = decodeSingleTokensPerSecond * kvBytesPerToken * 2;
  const decodeBandwidth = decodeWeightBandwidth + decodeKvBandwidth;
  const computeTops = (2 * params * decodeAggregateTokensPerSecond) / 1e12;
  const storageBytes = Math.max(metadata?.storageBytes || 0, weightBytes) * 1.15;

  const formulas: FormulaRow[] = [
    {
      item: "权重容量",
      level: "key",
      formula: "参数量 x 权重字节数",
      inputs: `${formatNumber(params / 1e9)}B params, ${precision.weights}`,
      result: formatGib(weightBytes)
    },
    {
      item: "KV cache",
      level: "key",
      formula: "batch x context x layers x 2 x kv_heads x head_dim x KV字节数",
      inputs: `${workload.batchSize} x ${context} x ${layers} x 2 x ${kvHeads} x ${formatNumber(headDim)} x ${precision.kvCache}`,
      result: formatGib(kvBytes)
    },
    {
      item: "Prefill 计算量",
      formula: "2 x 参数量 x batch x prompt_tokens + attention二次项",
      inputs: `${formatNumber(params / 1e9)}B params, batch ${workload.batchSize}, prompt ${workload.promptTokens}`,
      result: formatComputeOps(prefillOps)
    },
    {
      item: "Prefill 算力",
      level: "key",
      formula: "Prefill 计算量 / 目标 Prefill 时间",
      inputs: `${formatComputeOps(prefillOps)}, ${workload.targetPrefillMs} ms`,
      result: formatComputeTops(prefillComputeTops)
    },
    {
      item: "Decode 聚合吞吐",
      formula: "单路 TG token/s x batch",
      inputs: `${decodeSingleTokensPerSecond} x ${workload.batchSize}`,
      result: `${formatNumber(decodeAggregateTokensPerSecond)} tokens/s`
    },
    {
      item: "Decode 权重带宽",
      level: "key",
      formula: "聚合 tokens/s x 有效权重读取量",
      inputs: `${formatNumber(decodeAggregateTokensPerSecond)} tokens/s, batch ${workload.batchSize}`,
      result: `${formatNumber(decodeWeightBandwidth / GB)} GB/s`
    },
    {
      item: "Decode KV 带宽",
      level: "key",
      formula: "单路 TG token/s x batch x layers x 2 x kv_heads x head_dim x KV字节数 x 读写系数",
      inputs: `${decodeSingleTokensPerSecond} tokens/s, batch ${workload.batchSize}, ${precision.kvCache}`,
      result: `${formatNumber(decodeKvBandwidth / GB)} GB/s`
    },
    {
      item: "Decode 总带宽",
      level: "key",
      formula: "Decode 权重带宽 + Decode KV 带宽",
      inputs: `${formatNumber(decodeWeightBandwidth / GB)} + ${formatNumber(decodeKvBandwidth / GB)}`,
      result: `${formatNumber(decodeBandwidth / GB)} GB/s`
    }
  ];

  return {
    title: `${metadata?.name || "LLM"} 推理硬件需求`,
    kind: "llm",
    confidence: explicitParams && metadata?.config ? "medium" : "low",
    summary: `典型配置需要约 ${formatNumber(gb(totalMemory))} GB 内存；Prefill 在 ${formatNumber(
      workload.targetPrefillMs
    )} ms 目标下约需 ${formatComputeTops(prefillComputeTops)}，Decode 总带宽约 ${formatNumber(
      decodeBandwidth / GB
    )} GB/s。`,
    metrics: [
      metric("总内存容量", "GB", gb(totalMemory), "含运行时安全余量"),
      metric("模型存储容量", "GB", gb(storageBytes), "模型文件、tokenizer 与部署文件"),
      metric("Prefill 计算量", "Op", prefillOps, "batch x prompt tokens 的总操作量", 1, 1),
      metric("目标 Prefill 时间", "ms", workload.targetPrefillMs, "用于估算首 token 前的算力需求", 1, 1),
      metric("Prefill TOPS", "TOPS", prefillComputeTops, "按 batch x prompt tokens x 目标 TTFT 反推"),
      metric("Decode 单路 TG", "tokens/s", decodeSingleTokensPerSecond, "单个请求的生成速度", 1, 1),
      metric("Decode 聚合 TG", "tokens/s", decodeAggregateTokensPerSecond, "单路 TG x batch", 1, 1),
      metric("Decode 权重带宽", "GB/s", decodeWeightBandwidth / GB, "聚合 TG x 有效权重读取量"),
      metric("Decode KV 带宽", "GB/s", decodeKvBandwidth / GB, "KV cache 读写估算"),
      metric("Decode 总带宽", "GB/s", decodeBandwidth / GB, "权重带宽 + KV 带宽"),
      metric("Decode TOPS", "TOPS", computeTops, "按 2 x 参数量 x 聚合 TG 粗估")
    ],
    memoryBreakdown: memoryItems,
    formulas,
    assumptions: [
      "Prefill 近似为 2 x 参数量 x prompt tokens，并加入 attention 的二次项；它主要用于估算 TTFT 算力压力。",
      "Decode TG 在简单模式下表示单路 token/s，报告会按 batch 自动换算聚合 token/s。",
      "Decode 总带宽拆为权重带宽和 KV 带宽；batch 越大，权重读取摊销越明显。",
      "Decode KV 带宽按每个新 token 的 KV 读写近似估算，当前读写系数取 2。",
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
    summary: `典型配置需要约 ${formatNumber(gb(totalMemory))} GB 内存，目标实时率下约 ${formatComputeTops(
      computeTops
    )}。`,
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
        result: formatComputeTops(computeTops)
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
    summary: `典型配置需要约 ${formatNumber(gb(totalMemory))} GB 内存，vocoder 侧实时生成约需 ${formatComputeTops(
      computeTops
    )}。`,
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
        result: formatComputeTops(computeTops)
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
      ? 1
      : Math.round((config.imageSize / 8) ** 2 + (config.imageSize / 16) ** 2 + (config.imageSize / 32) ** 2);
  const outputValues = config.task === "classify" ? config.classes : config.classes + task.outputExtraValues;
  const outputBytes = config.batchSize * candidateCount * outputValues * precisionBytes;
  const postprocessBytes =
    config.task === "classify" ? config.batchSize * outputValues * 4 : config.batchSize * candidateCount * outputValues * 4 * 1.8;
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
  const storageBytes = weightBytes * 1.25;
  const bandwidthPerBatchBytes = weightBytes * 0.18 + activationBytes * 3.2 + outputBytes * 2.4 + postprocessBytes * 1.1;
  const bandwidthBytes = config.targetFps * bandwidthPerBatchBytes;

  return {
    title: `${scale.label} ${task.label} 推理硬件需求`,
    kind: "yolo",
    confidence: "medium",
    summary: `在 ${config.imageSize}x${config.imageSize}、单路 ${config.targetFps} FPS、batch ${config.batchSize} 下，典型需要约 ${formatNumber(
      gb(totalMemory)
    )} GB 运行内存和 ${formatComputeTops(requiredTops)}。`,
    metrics: [
      metric("总内存容量", "GB", gb(totalMemory), "含特征图、输出和后处理工作区"),
      metric("模型存储容量", "GB", gb(storageBytes), "权重与部署文件"),
      metric("主网络算力", "TOPS", requiredTops, `按 ${config.backend} 有效利用率估算`),
      metric("DDR/显存带宽", "GB/s", bandwidthBytes / GB, "单 batch 访问量 x 单路 FPS"),
      metric("单路帧率", "FPS", config.targetFps, "每路输入流目标帧率；batch 表示并行路数", 1, 1),
      metric("后处理工作区", "MB", mb(postprocessBytes), task.postprocessLabel)
    ],
    memoryBreakdown: memoryItems,
    formulas: [
      {
        item: "参数量",
        formula: "基准参数量 x 任务参数系数 x 类别head修正",
        inputs: `${scale.paramsM}M x ${formatNumber(task.paramsMultiplier)} x ${formatNumber(headAdjustment)}`,
        result: `${formatNumber(params / 1e6)}M params`
      },
      {
        item: "模型权重",
        formula: "参数量 x 精度字节数",
        inputs: `${formatNumber(params / 1e6)}M params x ${precisionBytes} bytes (${config.precision})`,
        result: formatGib(weightBytes)
      },
      {
        item: "模型存储",
        formula: "模型权重 x 部署文件系数",
        inputs: `${formatBytes(weightBytes)} x 1.25`,
        result: formatBytes(storageBytes)
      },
      {
        item: "输入尺寸系数",
        formula: "(输入尺寸 / 640)^2",
        inputs: `(${config.imageSize} / 640)^2`,
        result: formatNumber(sizeFactor)
      },
      {
        item: "YOLO FLOPs",
        formula: "640基准GFLOPs x 输入尺寸系数 x 任务FLOPs系数 x 类别head修正",
        inputs: `${scale.gflops640} x ${formatNumber(sizeFactor)} x ${formatNumber(task.flopsMultiplier)} x ${formatNumber(
          headAdjustment
        )}`,
        result: `${formatNumber(gflopsPerFrame)} GFLOPs/frame`
      },
      {
        item: "实时算力",
        level: "key",
        formula: "GFLOPs/frame x 单路 FPS x batch / 后端有效利用率",
        inputs: `${formatNumber(gflopsPerFrame)} x ${config.targetFps} x ${config.batchSize} / ${formatNumber(
          backendEfficiency
        )}`,
        result: formatComputeTops(requiredTops)
      },
      {
        item: "特征图峰值",
        formula: "batch x image_size x image_size x 精度字节数 x 任务激活系数 x 任务FLOPs系数",
        inputs: `${config.batchSize} x ${config.imageSize} x ${config.imageSize} x ${precisionBytes} x ${formatNumber(
          task.activationFactor
        )} x ${formatNumber(task.flopsMultiplier)}`,
        result: formatGib(activationBytes)
      },
      {
        item: "候选数",
        formula: "classify: 1；detect/segment/pose/obb: (S/8)^2 + (S/16)^2 + (S/32)^2",
        inputs:
          config.task === "classify"
            ? "1 classification output"
            : `(${config.imageSize}/8)^2 + (${config.imageSize}/16)^2 + (${config.imageSize}/32)^2`,
        result: `${candidateCount} candidates`
      },
      {
        item: "输出字段数",
        formula: "classify: 类别数；其他任务: 类别数 + 任务额外字段",
        inputs: `${config.classes} classes + ${task.outputExtraValues} extra`,
        result: `${outputValues} values`
      },
      {
        item: "候选框/输出",
        formula: "batch x 候选数 x 输出字段数 x 精度字节数",
        inputs: `${config.batchSize} x ${candidateCount} x ${outputValues} x ${precisionBytes}`,
        result: formatBytes(outputBytes)
      },
      {
        item: "后处理工作区",
        level: "key",
        formula: "classify: batch x 类别数 x 4；其他任务: batch x 候选数 x 输出字段数 x fp32字节数 x NMS系数",
        inputs:
          config.task === "classify"
            ? `${config.batchSize} x ${outputValues} x 4`
            : `${config.batchSize} x ${candidateCount} x ${outputValues} x 4 x 1.8`,
        result: formatBytes(postprocessBytes)
      },
      {
        item: "运行时余量",
        formula: "max(256MB, 权重 x 0.18 + 特征图 x 0.45 + 输出tensor)",
        inputs: `max(256MB, ${formatBytes(weightBytes)} x 0.18 + ${formatBytes(activationBytes)} x 0.45 + ${formatBytes(
          outputBytes
        )})`,
        result: formatGib(runtimeBytes)
      },
      {
        item: "总内存容量",
        level: "key",
        formula: "模型权重 + 特征图峰值 + 输出tensor + 后处理工作区 + 运行时余量",
        inputs: `${formatGib(weightBytes)} + ${formatGib(activationBytes)} + ${formatGib(outputBytes)} + ${formatGib(
          postprocessBytes
        )} + ${formatGib(runtimeBytes)}`,
        result: formatBytes(totalMemory)
      },
      {
        item: "单 batch 带宽访问量",
        formula: "权重 x 0.18 + 特征图 x 3.2 + 输出tensor x 2.4 + 后处理工作区 x 1.1",
        inputs: `${formatBytes(weightBytes)} x 0.18 + ${formatBytes(activationBytes)} x 3.2 + ${formatBytes(
          outputBytes
        )} x 2.4 + ${formatBytes(postprocessBytes)} x 1.1`,
        result: formatBytes(bandwidthPerBatchBytes)
      },
      {
        item: "DDR/显存带宽",
        level: "key",
        formula: "单路 FPS x 单 batch 带宽访问量",
        inputs: `${config.targetFps} x ${formatBytes(bandwidthPerBatchBytes)}`,
        result: `${formatNumber(bandwidthBytes / GB)} GB/s`
      }
    ],
    assumptions: [
      "YOLO 目录使用 Ultralytics 当前主线模型规模，任务 head 通过系数调整。",
      "YOLO 的特征图、输出和后处理工作区已包含 batch，带宽按单 batch 访问量 x 单路 FPS 估算。",
      "后处理单独估算，Detection/Segmentation/Pose/OBB 与 Classification 的工作区不同。",
      "自定义类别数主要影响 head、输出 tensor 和后处理，不重估完整 backbone。"
    ],
    warnings: [
      "第一版不解析 .pt、YAML 或 ONNX，实际自定义模型请用实测 FLOPs 校准。",
      "NMS 在 CPU 或低端 NPU 上可能成为延迟瓶颈，评审时应单独验证。"
    ]
  };
}
