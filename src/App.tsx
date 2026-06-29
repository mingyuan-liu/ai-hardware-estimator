import { useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  Cpu,
  Database,
  Download,
  FileJson,
  Gauge,
  RefreshCw,
  Search,
  Wand2
} from "lucide-react";
import { yoloScales, yoloTasks } from "./lib/catalog";
import { estimateRepositoryModel, estimateYolo } from "./lib/estimators";
import { confidenceLabel, formatBand, formatBytes, formatGib, reportToMarkdown } from "./lib/format";
import { inferModelKind } from "./lib/model";
import type {
  ModelKind,
  ModelMetadata,
  ModelSource,
  Precision,
  PrecisionConfig,
  RepositoryWorkload,
  RequirementReport,
  YoloConfig
} from "./lib/types";

const kindOptions: Array<{ value: ModelKind; label: string }> = [
  { value: "llm", label: "LLM" },
  { value: "asr", label: "ASR" },
  { value: "tts", label: "TTS" },
  { value: "yolo", label: "YOLO" }
];

const precisionOptions: Precision[] = ["fp32", "bf16", "fp16", "int8", "int4"];

interface CatalogModel {
  id: string;
  kind: ModelKind;
  vendor: string;
  name: string;
  description: string;
  source: ModelSource;
  modelId: string;
  yolo?: Pick<YoloConfig, "modelId" | "task">;
}

const repositoryCatalog: CatalogModel[] = [
  {
    id: "qwen-qwen3-0.6b-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen3-0.6B",
    description: "小型 LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen3-0.6B"
  },
  {
    id: "qwen-qwen3-1.7b-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen3-1.7B",
    description: "小型 LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen3-1.7B"
  },
  {
    id: "qwen-qwen3-4b-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen3-4B",
    description: "中小型 LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen3-4B"
  },
  {
    id: "qwen-qwen3-8b-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen3-8B",
    description: "通用 LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen3-8B"
  },
  {
    id: "qwen-qwen3-14b-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen3-14B",
    description: "中型 LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen3-14B"
  },
  {
    id: "qwen-qwen3-32b-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen3-32B",
    description: "大型 LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen3-32B"
  },
  {
    id: "qwen-qwen3-30b-a3b-moe-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen3-30B-A3B",
    description: "MoE LLM，总参数 30B、激活约 3B，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen3-30B-A3B"
  },
  {
    id: "qwen-qwen3-235b-a22b-moe-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen3-235B-A22B",
    description: "大型 MoE LLM，总参数 235B、激活约 22B，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen3-235B-A22B"
  },
  {
    id: "qwen-qwen1.5-moe-a2.7b-chat",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen1.5-MoE-A2.7B-Chat",
    description: "早期 Qwen MoE 对话模型，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "Qwen/Qwen1.5-MoE-A2.7B-Chat"
  },
  {
    id: "qwen-qwen2.5-0.5b-hf",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen2.5-0.5B-Instruct",
    description: "轻量指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "Qwen/Qwen2.5-0.5B-Instruct"
  },
  {
    id: "qwen-qwen2.5-1.5b-hf",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen2.5-1.5B-Instruct",
    description: "小型指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "Qwen/Qwen2.5-1.5B-Instruct"
  },
  {
    id: "qwen-qwen2.5-3b-hf",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen2.5-3B-Instruct",
    description: "小型指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "Qwen/Qwen2.5-3B-Instruct"
  },
  {
    id: "qwen-qwen2.5-7b-hf",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen2.5-7B-Instruct",
    description: "通用指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "Qwen/Qwen2.5-7B-Instruct"
  },
  {
    id: "qwen-qwen2.5-14b-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen2.5-14B-Instruct",
    description: "中型指令 LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen2.5-14B-Instruct"
  },
  {
    id: "qwen-qwen2.5-32b-ms",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen2.5-32B-Instruct",
    description: "大型指令 LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "Qwen/Qwen2.5-32B-Instruct"
  },
  {
    id: "qwen-qwen2.5-72b-hf",
    kind: "llm",
    vendor: "Qwen",
    name: "Qwen2.5-72B-Instruct",
    description: "超大指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "Qwen/Qwen2.5-72B-Instruct"
  },
  {
    id: "deepseek-r1-distill-qwen-1.5b",
    kind: "llm",
    vendor: "DeepSeek",
    name: "DeepSeek-R1-Distill-Qwen-1.5B",
    description: "小型推理蒸馏 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  },
  {
    id: "deepseek-r1-distill-qwen-7b",
    kind: "llm",
    vendor: "DeepSeek",
    name: "DeepSeek-R1-Distill-Qwen-7B",
    description: "推理蒸馏 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  },
  {
    id: "deepseek-r1-distill-qwen-14b",
    kind: "llm",
    vendor: "DeepSeek",
    name: "DeepSeek-R1-Distill-Qwen-14B",
    description: "推理蒸馏 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  },
  {
    id: "deepseek-r1-distill-qwen-32b",
    kind: "llm",
    vendor: "DeepSeek",
    name: "DeepSeek-R1-Distill-Qwen-32B",
    description: "大型推理蒸馏 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  },
  {
    id: "deepseek-r1-distill-llama-8b",
    kind: "llm",
    vendor: "DeepSeek",
    name: "DeepSeek-R1-Distill-Llama-8B",
    description: "推理蒸馏 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  },
  {
    id: "deepseek-v2-lite-chat-moe-ms",
    kind: "llm",
    vendor: "DeepSeek",
    name: "DeepSeek-V2-Lite-Chat",
    description: "轻量 MoE 对话 LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "deepseek-ai/DeepSeek-V2-Lite-Chat"
  },
  {
    id: "deepseek-v2-chat-moe",
    kind: "llm",
    vendor: "DeepSeek",
    name: "DeepSeek-V2-Chat",
    description: "MoE 对话 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "deepseek-ai/DeepSeek-V2-Chat"
  },
  {
    id: "deepseek-v3-moe-ms",
    kind: "llm",
    vendor: "DeepSeek",
    name: "DeepSeek-V3",
    description: "大型 MoE LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "deepseek-ai/DeepSeek-V3"
  },
  {
    id: "deepseek-r1-moe-ms",
    kind: "llm",
    vendor: "DeepSeek",
    name: "DeepSeek-R1",
    description: "大型推理 MoE LLM，ModelScope 公开模型",
    source: "modelscope",
    modelId: "deepseek-ai/DeepSeek-R1"
  },
  {
    id: "microsoft-phi-4-mini",
    kind: "llm",
    vendor: "Microsoft",
    name: "Phi-4-mini-instruct",
    description: "小型指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "microsoft/Phi-4-mini-instruct"
  },
  {
    id: "microsoft-phi-3.5-mini",
    kind: "llm",
    vendor: "Microsoft",
    name: "Phi-3.5-mini-instruct",
    description: "小型指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "microsoft/Phi-3.5-mini-instruct"
  },
  {
    id: "microsoft-phi-3-mini",
    kind: "llm",
    vendor: "Microsoft",
    name: "Phi-3-mini-4k-instruct",
    description: "小型指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "microsoft/Phi-3-mini-4k-instruct"
  },
  {
    id: "mistral-7b-instruct-v0.3",
    kind: "llm",
    vendor: "Mistral AI",
    name: "Mistral-7B-Instruct-v0.3",
    description: "通用指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "mistralai/Mistral-7B-Instruct-v0.3"
  },
  {
    id: "mixtral-8x7b-instruct",
    kind: "llm",
    vendor: "Mistral AI",
    name: "Mixtral-8x7B-Instruct-v0.1",
    description: "MoE 指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "mistralai/Mixtral-8x7B-Instruct-v0.1"
  },
  {
    id: "mixtral-8x22b-instruct",
    kind: "llm",
    vendor: "Mistral AI",
    name: "Mixtral-8x22B-Instruct-v0.1",
    description: "大型 MoE 指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "mistralai/Mixtral-8x22B-Instruct-v0.1"
  },
  {
    id: "thudm-glm-4-9b-chat",
    kind: "llm",
    vendor: "THUDM",
    name: "GLM-4-9B-Chat",
    description: "中文/通用对话 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "THUDM/glm-4-9b-chat"
  },
  {
    id: "internlm2.5-7b-chat",
    kind: "llm",
    vendor: "InternLM",
    name: "InternLM2.5-7B-Chat",
    description: "通用对话 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "internlm/internlm2_5-7b-chat"
  },
  {
    id: "openbmb-minicpm3-4b",
    kind: "llm",
    vendor: "OpenBMB",
    name: "MiniCPM3-4B",
    description: "端侧友好 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "openbmb/MiniCPM3-4B"
  },
  {
    id: "openbmb-minicpm-moe-8x2b",
    kind: "llm",
    vendor: "OpenBMB",
    name: "MiniCPM-MoE-8x2B",
    description: "端侧 MoE LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "openbmb/MiniCPM-MoE-8x2B"
  },
  {
    id: "tinyllama-1.1b-chat",
    kind: "llm",
    vendor: "TinyLlama",
    name: "TinyLlama-1.1B-Chat",
    description: "轻量 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  },
  {
    id: "aidc-marco-o1-moe",
    kind: "llm",
    vendor: "AIDC",
    name: "Marco-o1",
    description: "推理 MoE LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "AIDC-AI/Marco-o1"
  },
  {
    id: "allenai-olmoe-1b-7b-instruct",
    kind: "llm",
    vendor: "AllenAI",
    name: "OLMoE-1B-7B-0924-Instruct",
    description: "开放 MoE 指令 LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "allenai/OLMoE-1B-7B-0924-Instruct"
  },
  {
    id: "snowflake-arctic-instruct",
    kind: "llm",
    vendor: "Snowflake",
    name: "snowflake-arctic-instruct",
    description: "企业场景 MoE LLM，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "Snowflake/snowflake-arctic-instruct"
  },
  {
    id: "openai-whisper-tiny",
    kind: "asr",
    vendor: "OpenAI",
    name: "Whisper Tiny",
    description: "轻量 ASR，适合边缘估算",
    source: "huggingface",
    modelId: "openai/whisper-tiny"
  },
  {
    id: "openai-whisper-base",
    kind: "asr",
    vendor: "OpenAI",
    name: "Whisper Base",
    description: "基础 ASR，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "openai/whisper-base"
  },
  {
    id: "openai-whisper-small",
    kind: "asr",
    vendor: "OpenAI",
    name: "Whisper Small",
    description: "常用 ASR 基准模型",
    source: "huggingface",
    modelId: "openai/whisper-small"
  },
  {
    id: "openai-whisper-medium",
    kind: "asr",
    vendor: "OpenAI",
    name: "Whisper Medium",
    description: "中型 ASR，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "openai/whisper-medium"
  },
  {
    id: "openai-whisper-large-v3",
    kind: "asr",
    vendor: "OpenAI",
    name: "Whisper Large v3",
    description: "大型 ASR，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "openai/whisper-large-v3"
  },
  {
    id: "openai-whisper-large-v3-turbo",
    kind: "asr",
    vendor: "OpenAI",
    name: "Whisper Large v3 Turbo",
    description: "大型快速 ASR，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "openai/whisper-large-v3-turbo"
  },
  {
    id: "distil-whisper-large-v3",
    kind: "asr",
    vendor: "Distil-Whisper",
    name: "Distil-Whisper Large v3",
    description: "蒸馏 ASR，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "distil-whisper/distil-large-v3"
  },
  {
    id: "facebook-wav2vec2-base-960h",
    kind: "asr",
    vendor: "Meta / Facebook",
    name: "Wav2Vec2 Base 960h",
    description: "经典 CTC ASR，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "facebook/wav2vec2-base-960h"
  },
  {
    id: "facebook-hubert-large-ls960",
    kind: "asr",
    vendor: "Meta / Facebook",
    name: "HuBERT Large LS960",
    description: "经典语音表征/ASR，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "facebook/hubert-large-ls960-ft"
  },
  {
    id: "funaudio-sensevoice-small-hf",
    kind: "asr",
    vendor: "FunAudioLLM",
    name: "SenseVoiceSmall",
    description: "多语种 ASR，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "FunAudioLLM/SenseVoiceSmall"
  },
  {
    id: "alibaba-sensevoice-small",
    kind: "asr",
    vendor: "Alibaba / iic",
    name: "SenseVoiceSmall",
    description: "中文语音识别，ModelScope 公开模型",
    source: "modelscope",
    modelId: "iic/SenseVoiceSmall"
  },
  {
    id: "alibaba-paraformer-large",
    kind: "asr",
    vendor: "Alibaba / iic",
    name: "Paraformer Large 中文 ASR",
    description: "中文 ASR，ModelScope 公开模型",
    source: "modelscope",
    modelId: "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
  },
  {
    id: "alibaba-fsmn-vad",
    kind: "asr",
    vendor: "Alibaba / iic",
    name: "FSMN VAD 中文 16k",
    description: "语音端点检测，ModelScope 公开模型",
    source: "modelscope",
    modelId: "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
  },
  {
    id: "microsoft-speecht5-tts",
    kind: "tts",
    vendor: "Microsoft",
    name: "SpeechT5 TTS",
    description: "文本转语音，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "microsoft/speecht5_tts"
  },
  {
    id: "suno-bark-small",
    kind: "tts",
    vendor: "Suno",
    name: "Bark Small",
    description: "生成式 TTS，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "suno/bark-small"
  },
  {
    id: "facebook-mms-tts-eng",
    kind: "tts",
    vendor: "Meta / Facebook",
    name: "MMS TTS English",
    description: "多语种 TTS 系列英文模型，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "facebook/mms-tts-eng"
  },
  {
    id: "hexgrad-kokoro-82m",
    kind: "tts",
    vendor: "Kokoro",
    name: "Kokoro-82M",
    description: "轻量 TTS，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "hexgrad/Kokoro-82M"
  },
  {
    id: "parler-tts-mini",
    kind: "tts",
    vendor: "Parler-TTS",
    name: "Parler-TTS Mini v1",
    description: "文本提示控制 TTS，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "parler-tts/parler-tts-mini-v1"
  },
  {
    id: "espnet-ljspeech-vits",
    kind: "tts",
    vendor: "ESPnet",
    name: "LJSpeech VITS",
    description: "VITS TTS，Hugging Face 公开模型",
    source: "huggingface",
    modelId: "espnet/kan-bayashi_ljspeech_vits"
  }
];

const yoloCatalog: CatalogModel[] = yoloScales.flatMap((scale) =>
  yoloTasks.map((task) => ({
    id: `${scale.id}-${task.id}`,
    kind: "yolo" as const,
    vendor: "Ultralytics",
    name: `${scale.label} ${task.label}`,
    description: `${task.postprocessLabel}，${scale.paramsM}M 参数基准`,
    source: "huggingface" as const,
    modelId: `${scale.id}-${task.id}`,
    yolo: {
      modelId: scale.id,
      task: task.id
    }
  }))
);

const modelCatalog: CatalogModel[] = [...repositoryCatalog, ...yoloCatalog];
const initialCatalogModel = modelCatalog[0];

const defaultWorkload: RepositoryWorkload = {
  batchSize: 1,
  benchmarkTokens: 128,
  prefillTokensPerSecond: 56.43,
  promptTokens: 128,
  targetPrefillMs: 2268,
  contextTokens: 4096,
  outputTokens: 128,
  targetTokensPerSecond: 11.61,
  audioStreams: 4,
  targetRtf: 0.5,
  sampleRate: 16000,
  imageSize: 640,
  targetFps: 30
};

const defaultPrecision: PrecisionConfig = {
  weights: "fp16",
  activation: "fp16",
  kvCache: "fp16"
};

const defaultYolo: YoloConfig = {
  modelId: "yolo11n",
  task: "detect",
  imageSize: 640,
  batchSize: 1,
  targetFps: 30,
  classes: 80,
  precision: "int8",
  backend: "tensorrt"
};

function App() {
  const [kind, setKind] = useState<ModelKind>(initialCatalogModel.kind);
  const [selectedVendor, setSelectedVendor] = useState(initialCatalogModel.vendor);
  const [selectedCatalogId, setSelectedCatalogId] = useState(initialCatalogModel.id);
  const [source, setSource] = useState<ModelSource>(initialCatalogModel.source);
  const [modelId, setModelId] = useState(initialCatalogModel.modelId);
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [workload, setWorkload] = useState<RepositoryWorkload>(defaultWorkload);
  const [precision, setPrecision] = useState<PrecisionConfig>(defaultPrecision);
  const [yolo, setYolo] = useState<YoloConfig>(defaultYolo);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");

  const vendors = useMemo(
    () => Array.from(new Set(modelCatalog.filter((item) => item.kind === kind).map((item) => item.vendor))),
    [kind]
  );
  const selectableModels = useMemo(
    () => modelCatalog.filter((item) => item.kind === kind && item.vendor === selectedVendor),
    [kind, selectedVendor]
  );
  const selectedCatalogModel = modelCatalog.find((item) => item.id === selectedCatalogId) || selectableModels[0];

  const report = useMemo<RequirementReport>(() => {
    if (kind === "yolo") return estimateYolo(yolo);
    return estimateRepositoryModel(kind, metadata, workload, precision);
  }, [kind, metadata, precision, workload, yolo]);

  function applyCatalogModel(item: CatalogModel) {
    setKind(item.kind);
    setSelectedVendor(item.vendor);
    setSelectedCatalogId(item.id);
    setSource(item.source);
    setModelId(item.modelId);
    setLoadError("");
    if (item.kind !== "yolo") {
      setMetadata(null);
      return;
    }
    if (item.yolo) {
      setYolo((current) => ({
        ...current,
        modelId: item.yolo?.modelId || current.modelId,
        task: item.yolo?.task || current.task
      }));
    }
  }

  function handleKindChange(nextKind: ModelKind) {
    const first = modelCatalog.find((item) => item.kind === nextKind);
    if (first) applyCatalogModel(first);
  }

  function handleVendorChange(vendor: string) {
    const first = modelCatalog.find((item) => item.kind === kind && item.vendor === vendor);
    if (first) applyCatalogModel(first);
  }

  function handleCatalogModelChange(id: string) {
    const next = modelCatalog.find((item) => item.id === id);
    if (next) applyCatalogModel(next);
  }

  async function loadModel() {
    const parsed = parseModelInput(modelId, source);
    if (parsed.modelId !== modelId || parsed.source !== source) {
      setModelId(parsed.modelId);
      setSource(parsed.source);
    }
    setLoading(true);
    setLoadError("");
    try {
      const apiUrl = `${import.meta.env.BASE_URL}api/model?source=${parsed.source}&modelId=${encodeURIComponent(
        parsed.modelId
      )}`;
      const response = await fetch(
        apiUrl
      );
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "模型元数据获取失败");
      setMetadata(payload);
      const inferred = inferModelKind(payload);
      if (inferred !== "yolo") setKind(inferred);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "模型元数据获取失败");
    } finally {
      setLoading(false);
    }
  }

  function downloadReport(format: "json" | "md") {
    const content =
      format === "json" ? JSON.stringify(report, null, 2) : reportToMarkdown(report);
    const type = format === "json" ? "application/json" : "text/markdown";
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `hardware-requirement-${kind}.${format}`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Cpu size={24} aria-hidden="true" />
          <div>
            <h1>AI 硬件需求评估</h1>
            <p>模型元数据、推理负载与硬件需求报告</p>
          </div>
        </div>
        <div className="topbar-actions">
          <button className="icon-button" title="导出 JSON" onClick={() => downloadReport("json")}>
            <FileJson size={18} aria-hidden="true" />
            JSON
          </button>
          <button className="icon-button primary" title="导出 Markdown" onClick={() => downloadReport("md")}>
            <Download size={18} aria-hidden="true" />
            Markdown
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="control-panel">
          <section className="panel-section">
            <SectionHeader icon={<Database size={18} />} title="模型" />
            <SegmentedControl
              value={kind}
              options={kindOptions}
              onChange={(value) => handleKindChange(value as ModelKind)}
            />
            <ModelControls
              kind={kind}
              vendors={vendors}
              selectedVendor={selectedVendor}
              selectedModelId={selectedCatalogId}
              models={selectableModels}
              selectedModel={selectedCatalogModel}
              source={source}
              setSource={setSource}
              modelId={modelId}
              setModelId={setModelId}
              loading={loading}
              loadError={loadError}
              metadata={metadata}
              onVendorChange={handleVendorChange}
              onModelChange={handleCatalogModelChange}
              onLoad={loadModel}
            />
          </section>

          <section className="panel-section">
            <SectionHeader icon={<Gauge size={18} />} title="推理负载" />
            {kind === "yolo" ? (
              <YoloDeploymentControls yolo={yolo} setYolo={setYolo} />
            ) : (
              <RepositoryWorkloadControls kind={kind} workload={workload} setWorkload={setWorkload} />
            )}
          </section>
          {kind !== "yolo" && (
            <section className="panel-section">
              <SectionHeader icon={<BarChart3 size={18} />} title="精度" />
              <PrecisionControls kind={kind} precision={precision} setPrecision={setPrecision} />
            </section>
          )}
        </aside>

        <ReportView report={report} />
      </section>
    </main>
  );
}

function parseModelInput(input: string, currentSource: ModelSource): { source: ModelSource; modelId: string } {
  const trimmed = input.trim();
  if (!trimmed) return { source: currentSource, modelId: "" };

  try {
    const url = new URL(trimmed);
    const parts = url.pathname.split("/").filter(Boolean);
    if (url.hostname.includes("huggingface.co") && parts.length >= 2) {
      return { source: "huggingface", modelId: `${parts[0]}/${parts[1]}` };
    }
    if (url.hostname.includes("modelscope.cn")) {
      const modelIndex = parts.findIndex((part) => part === "models");
      const offset = modelIndex >= 0 ? modelIndex + 1 : 0;
      if (parts.length >= offset + 2) {
        return { source: "modelscope", modelId: `${parts[offset]}/${parts[offset + 1]}` };
      }
    }
  } catch {
    // Plain model IDs are handled below.
  }

  return {
    source: currentSource,
    modelId: trimmed.replace(/^\/+|\/+$/g, "")
  };
}

function getModelSourceUrl(model: CatalogModel): string {
  if (model.kind === "yolo") {
    const family = model.yolo?.modelId.startsWith("yolov8") ? "yolov8" : "yolo11";
    return `https://docs.ultralytics.com/models/${family}/`;
  }
  if (model.source === "modelscope") return `https://modelscope.cn/models/${model.modelId}`;
  return `https://huggingface.co/${model.modelId}`;
}

interface SectionHeaderProps {
  icon: React.ReactNode;
  title: string;
}

function SectionHeader({ icon, title }: SectionHeaderProps) {
  return (
    <div className="section-header">
      {icon}
      <h2>{title}</h2>
    </div>
  );
}

interface SegmentedControlProps<T extends string> {
  value: T;
  options: Array<{ value: T; label: string }>;
  onChange: (value: T) => void;
}

function SegmentedControl<T extends string>({ value, options, onChange }: SegmentedControlProps<T>) {
  return (
    <div className="segmented" role="tablist">
      {options.map((option) => (
        <button
          key={option.value}
          className={option.value === value ? "active" : ""}
          onClick={() => onChange(option.value)}
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

interface ModelControlsProps {
  kind: ModelKind;
  vendors: string[];
  selectedVendor: string;
  selectedModelId: string;
  models: CatalogModel[];
  selectedModel?: CatalogModel;
  source: ModelSource;
  setSource: (source: ModelSource) => void;
  modelId: string;
  setModelId: (modelId: string) => void;
  loading: boolean;
  loadError: string;
  metadata: ModelMetadata | null;
  onVendorChange: (vendor: string) => void;
  onModelChange: (id: string) => void;
  onLoad: () => void;
}

function ModelControls({
  kind,
  vendors,
  selectedVendor,
  selectedModelId,
  models,
  selectedModel,
  source,
  setSource,
  modelId,
  setModelId,
  loading,
  loadError,
  metadata,
  onVendorChange,
  onModelChange,
  onLoad
}: ModelControlsProps) {
  return (
    <div className="control-stack">
      <div className="model-picker">
        <div className="quick-models-title">
          <Wand2 size={15} aria-hidden="true" />
          <span>内置模型列表</span>
        </div>
        <div className="form-grid">
          <SelectField label="厂商/组织" value={selectedVendor} options={vendors} onChange={onVendorChange} />
          <SelectField
            label="模型"
            value={selectedModelId}
            options={models.map((item) => item.id)}
            getLabel={(value) => models.find((item) => item.id === value)?.name || value}
            onChange={onModelChange}
          />
        </div>
        {selectedModel && (
          <div className="model-summary">
            <strong>{selectedModel.name}</strong>
            <span>
              {selectedModel.description} ·{" "}
              {kind === "yolo"
                ? "内置 YOLO 参数"
                : `${selectedModel.source === "huggingface" ? "Hugging Face" : "ModelScope"}: ${selectedModel.modelId}`}
            </span>
            <a href={getModelSourceUrl(selectedModel)} target="_blank" rel="noreferrer">
              查看源链接
            </a>
          </div>
        )}
      </div>
      {kind !== "yolo" && (
        <>
          <button className="icon-button primary full-width" type="button" onClick={onLoad} disabled={loading}>
            {loading ? <RefreshCw size={17} className="spin" /> : <Search size={17} />}
            加载模型信息
          </button>
          <details className="advanced-input">
            <summary>高级：手动输入模型链接或 ID</summary>
            <div className="control-stack">
              <SegmentedControl
                value={source}
                options={[
                  { value: "huggingface", label: "Hugging Face" },
                  { value: "modelscope", label: "ModelScope" }
                ]}
                onChange={setSource}
              />
              <label className="field">
                <span>模型链接 / ID</span>
                <div className="inline-input">
                  <input
                    value={modelId}
                    placeholder="可粘贴 https://huggingface.co/Qwen/Qwen3-0.6B"
                    onChange={(event) => {
                      const next = event.target.value;
                      setModelId(next);
                      const parsed = parseModelInput(next, source);
                      if (parsed.source !== source) setSource(parsed.source);
                    }}
                  />
                  <button className="icon-button" type="button" onClick={onLoad} disabled={loading}>
                    {loading ? <RefreshCw size={17} className="spin" /> : <Search size={17} />}
                    拉取
                  </button>
                </div>
              </label>
            </div>
          </details>
        </>
      )}
      {loadError && (
        <div className="notice error">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>{loadError}</span>
        </div>
      )}
      {metadata && <MetadataSummary metadata={metadata} />}
    </div>
  );
}

function MetadataSummary({ metadata }: { metadata: ModelMetadata }) {
  const modelType = String(metadata.config?.model_type || metadata.pipelineTag || "unknown");
  return (
    <dl className="metadata-list">
      <div>
        <dt>名称</dt>
        <dd>{metadata.name}</dd>
      </div>
      <div>
        <dt>类型</dt>
        <dd>{modelType}</dd>
      </div>
      <div>
        <dt>文件</dt>
        <dd>{metadata.files.length}</dd>
      </div>
      <div>
        <dt>存储</dt>
        <dd>{metadata.storageBytes ? formatBytes(metadata.storageBytes) : "-"}</dd>
      </div>
    </dl>
  );
}

interface RepositoryWorkloadControlsProps {
  kind: ModelKind;
  workload: RepositoryWorkload;
  setWorkload: (workload: RepositoryWorkload) => void;
}

function RepositoryWorkloadControls({ kind, workload, setWorkload }: RepositoryWorkloadControlsProps) {
  const update = (patch: Partial<RepositoryWorkload>) => setWorkload({ ...workload, ...patch });

  if (kind === "llm") {
    const applyBenchmark = (patch: Partial<RepositoryWorkload>) => {
      const next = { ...workload, ...patch };
      const benchmarkTokens = Math.max(1, next.benchmarkTokens);
      const pp = Math.max(0.01, next.prefillTokensPerSecond);
      const promptTokens = benchmarkTokens;
      const outputTokens = benchmarkTokens;
      const contextTokens = Math.max(4096, promptTokens + outputTokens, next.contextTokens);
      setWorkload({
        ...next,
        benchmarkTokens,
        promptTokens,
        outputTokens,
        contextTokens,
        targetPrefillMs: Math.round((benchmarkTokens / pp) * 1000)
      });
    };

    return (
      <div className="control-stack">
        <div className="field-group">
          <div className="field-group-title">Benchmark 简单模式</div>
          <div className="form-grid">
            <NumberField
              label="Benchmark 长度"
              suffix="tokens"
              value={workload.benchmarkTokens}
              min={1}
              step={128}
              onChange={(benchmarkTokens) => applyBenchmark({ benchmarkTokens })}
            />
            <NumberField
              label="PP token/s"
              value={workload.prefillTokensPerSecond}
              min={0.01}
              step={1}
              onChange={(prefillTokensPerSecond) => applyBenchmark({ prefillTokensPerSecond })}
            />
            <NumberField
              label="TG token/s"
              value={workload.targetTokensPerSecond}
              min={0.01}
              step={1}
              onChange={(targetTokensPerSecond) => applyBenchmark({ targetTokensPerSecond })}
            />
            <ReadOnlyField label="Prefill 时间" value={`${workload.targetPrefillMs} ms`} />
          </div>
        </div>
        <details className="advanced-input">
          <summary>高级：请求规模与上下文</summary>
          <div className="control-stack">
            <div className="form-grid">
              <NumberField
                label="Batch / 并发"
                value={workload.batchSize}
                min={1}
                step={1}
                onChange={(batchSize) => update({ batchSize })}
              />
              <NumberField
                label="Prompt tokens"
                value={workload.promptTokens}
                min={1}
                step={128}
                onChange={(promptTokens) => update({ promptTokens })}
              />
              <NumberField
                label="生成 tokens"
                value={workload.outputTokens}
                min={1}
                step={128}
                onChange={(outputTokens) => update({ outputTokens })}
              />
              <NumberField
                label="最大 Context tokens"
                value={workload.contextTokens}
                min={1}
                step={512}
                onChange={(contextTokens) => update({ contextTokens })}
              />
              <NumberField
                label="目标 Prefill 时间"
                suffix="ms"
                value={workload.targetPrefillMs}
                min={10}
                step={50}
                onChange={(targetPrefillMs) => update({ targetPrefillMs })}
              />
            </div>
          </div>
        </details>
      </div>
    );
  }

  return (
    <div className="form-grid">
      <NumberField
        label="并发流"
        value={workload.audioStreams}
        min={1}
        step={1}
        onChange={(audioStreams) => update({ audioStreams })}
      />
      <NumberField
        label="RTF 目标"
        value={workload.targetRtf}
        min={0.05}
        step={0.05}
        onChange={(targetRtf) => update({ targetRtf })}
      />
      <NumberField
        label="采样率"
        suffix="Hz"
        value={workload.sampleRate}
        min={8000}
        step={1000}
        onChange={(sampleRate) => update({ sampleRate })}
      />
    </div>
  );
}

function PrecisionControls({
  kind,
  precision,
  setPrecision
}: {
  kind: ModelKind;
  precision: PrecisionConfig;
  setPrecision: (precision: PrecisionConfig) => void;
}) {
  if (kind === "llm") {
    return (
      <div className="control-stack">
        <div className="form-grid">
          <SelectField
            label="权重量化"
            value={precision.weights}
            options={precisionOptions}
            onChange={(weights) => setPrecision({ ...precision, weights })}
          />
        </div>
        <details className="advanced-input">
          <summary>高级：运行精度</summary>
          <div className="control-stack">
            <div className="form-grid">
              <SelectField
                label="激活精度"
                value={precision.activation}
                options={precisionOptions}
                onChange={(activation) => setPrecision({ ...precision, activation })}
              />
              <SelectField
                label="KV Cache 精度"
                value={precision.kvCache}
                options={precisionOptions}
                onChange={(kvCache) => setPrecision({ ...precision, kvCache })}
              />
            </div>
          </div>
        </details>
      </div>
    );
  }

  return (
    <div className="form-grid">
      <SelectField
        label="权重量化"
        value={precision.weights}
        options={precisionOptions}
        onChange={(weights) => setPrecision({ ...precision, weights })}
      />
      <SelectField
        label="激活精度"
        value={precision.activation}
        options={precisionOptions}
        onChange={(activation) => setPrecision({ ...precision, activation })}
      />
      <SelectField
        label="KV Cache 精度"
        value={precision.kvCache}
        options={precisionOptions}
        onChange={(kvCache) => setPrecision({ ...precision, kvCache })}
      />
    </div>
  );
}

function YoloDeploymentControls({ yolo, setYolo }: { yolo: YoloConfig; setYolo: (yolo: YoloConfig) => void }) {
  const update = (patch: Partial<YoloConfig>) => setYolo({ ...yolo, ...patch });

  return (
    <div className="control-stack">
      <div className="form-grid">
        <NumberField label="输入尺寸" suffix="px" value={yolo.imageSize} min={160} step={32} onChange={(imageSize) => update({ imageSize })} />
        <NumberField label="Batch" value={yolo.batchSize} min={1} step={1} onChange={(batchSize) => update({ batchSize })} />
        <NumberField label="单路 FPS" suffix="FPS" value={yolo.targetFps} min={1} step={5} onChange={(targetFps) => update({ targetFps })} />
        <NumberField label="类别数" value={yolo.classes} min={1} step={1} onChange={(classes) => update({ classes })} />
        <SelectField label="精度" value={yolo.precision} options={precisionOptions} onChange={(value) => update({ precision: value })} />
        <SelectField
          label="后端"
          value={yolo.backend}
          options={["generic", "onnxruntime", "tensorrt", "npu"]}
          getLabel={(value) =>
            ({ generic: "通用", onnxruntime: "ONNX Runtime", tensorrt: "TensorRT", npu: "NPU" })[value]
          }
          onChange={(backend) => update({ backend })}
        />
      </div>
    </div>
  );
}

interface NumberFieldProps {
  label: string;
  value: number;
  min?: number;
  step?: number;
  suffix?: string;
  onChange: (value: number) => void;
}

function NumberField({ label, value, min, step, suffix, onChange }: NumberFieldProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="input-with-suffix">
        <input
          type="number"
          value={value}
          min={min}
          step={step}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        {suffix && <em>{suffix}</em>}
      </div>
    </label>
  );
}

function ReadOnlyField({ label, value }: { label: string; value: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      <output className="readonly-field">{value}</output>
    </label>
  );
}

interface SelectFieldProps<T extends string> {
  label: string;
  value: T;
  options: T[];
  getLabel?: (value: T) => string;
  onChange: (value: T) => void;
}

function SelectField<T extends string>({ label, value, options, getLabel, onChange }: SelectFieldProps<T>) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value as T)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {getLabel ? getLabel(option) : option.toUpperCase()}
          </option>
        ))}
      </select>
    </label>
  );
}

function ReportView({ report }: { report: RequirementReport }) {
  return (
    <section className="report-panel">
      <div className="report-header">
        <div>
          <p className="eyebrow">{report.kind.toUpperCase()} · 置信度 {confidenceLabel(report.confidence)}</p>
          <h2>{report.title}</h2>
          <p>{report.summary}</p>
        </div>
      </div>

      <MetricTable report={report} />
      <MemoryBreakdown report={report} />
      <FormulaTable report={report} />
      <AssumptionList report={report} />
      <UnitGuide />
    </section>
  );
}

function UnitGuide() {
  return (
    <section className="report-section compact-section">
      <h3>单位速查</h3>
      <div className="unit-grid">
        <div>
          <strong>GOp / TOp</strong>
          <span>总操作量；1 GOp = 10 亿次，1 TOp = 1 万亿次 = 1000 GOp。</span>
        </div>
        <div>
          <strong>GOPS / TOPS</strong>
          <span>每秒操作量；1 GOPS = 每秒 10 亿次，1 TOPS = 每秒 1 万亿次。</span>
        </div>
        <div>
          <strong>GB / GiB</strong>
          <span>同一容量的十进制/二进制表示；1 GiB = 1.074 GB。</span>
        </div>
        <div>
          <strong>GB/s</strong>
          <span>每秒数据搬运量，用来看内存或显存带宽。</span>
        </div>
      </div>
    </section>
  );
}

function MetricTable({ report }: { report: RequirementReport }) {
  return (
    <section className="report-section">
      <h3>需求指标</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>指标</th>
              <th>激进</th>
              <th>典型</th>
              <th>保守</th>
              <th>说明</th>
            </tr>
          </thead>
          <tbody>
            {report.metrics.map((metric) => (
              <tr key={metric.label}>
                <td>{metric.label}</td>
                <td>{formatBand(metric.value.aggressive, metric.unit)}</td>
                <td className="strong">{formatBand(metric.value.typical, metric.unit)}</td>
                <td>{formatBand(metric.value.conservative, metric.unit)}</td>
                <td>{metric.note || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MemoryBreakdown({ report }: { report: RequirementReport }) {
  const total = report.memoryBreakdown.reduce((sum, item) => sum + item.bytes, 0);
  return (
    <section className="report-section">
      <h3>内存构成</h3>
      <div className="memory-bar" aria-label="内存构成图">
        {report.memoryBreakdown.map((item) => (
          <span
            key={item.label}
            className={`memory-segment ${item.tone}`}
            style={{ width: `${Math.max(3, (item.bytes / total) * 100)}%` }}
            title={`${item.label}: ${formatGib(item.bytes)}`}
          />
        ))}
      </div>
      <div className="breakdown-grid">
        {report.memoryBreakdown.map((item) => (
          <div className="breakdown-row" key={item.label}>
            <span className={`dot ${item.tone}`} />
            <span>{item.label}</span>
            <strong>{formatGib(item.bytes)}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function FormulaTable({ report }: { report: RequirementReport }) {
  const keyFormulas = report.formulas.filter((row) => row.level === "key");
  const advancedFormulas = report.formulas.filter((row) => row.level !== "key");
  const visibleKeyFormulas = keyFormulas.length ? keyFormulas : report.formulas;

  return (
    <section className="report-section">
      <h3>关键公式</h3>
      <FormulaRows rows={visibleKeyFormulas} />
      {advancedFormulas.length > 0 && keyFormulas.length > 0 && (
        <details className="formula-details">
          <summary>高级公式 / 展开全部</summary>
          <FormulaRows rows={advancedFormulas} />
        </details>
      )}
    </section>
  );
}

function FormulaRows({ rows }: { rows: RequirementReport["formulas"] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>项目</th>
            <th>公式</th>
            <th>输入</th>
            <th>结果</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.item}>
              <td>{row.item}</td>
              <td>{row.formula}</td>
              <td>{row.inputs}</td>
              <td className="strong">{row.result}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AssumptionList({ report }: { report: RequirementReport }) {
  return (
    <section className="report-section split">
      <div>
        <h3>假设</h3>
        <ul>
          {report.assumptions.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>
      <div>
        <h3>风险</h3>
        <ul className="warning-list">
          {report.warnings.length ? report.warnings.map((item) => <li key={item}>{item}</li>) : <li>无</li>}
        </ul>
      </div>
    </section>
  );
}

export default App;
