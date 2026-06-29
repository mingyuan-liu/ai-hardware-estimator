import type { ModelKind, ModelMetadata } from "./types";

export function inferModelKind(metadata?: ModelMetadata | null): ModelKind {
  const config = metadata?.config || {};
  const modelType = String(config.model_type || "").toLowerCase();
  const arch = Array.isArray(config.architectures) ? config.architectures.join(" ").toLowerCase() : "";
  const tags = (metadata?.tags || []).join(" ").toLowerCase();
  const pipeline = String(metadata?.pipelineTag || "").toLowerCase();
  const haystack = `${modelType} ${arch} ${tags} ${pipeline}`;

  if (/yolo|object-detection|image-segmentation|pose|obb/.test(haystack)) return "yolo";
  if (/whisper|wav2vec|hubert|conformer|paraformer|asr|automatic-speech-recognition|speech-recognition/.test(haystack)) {
    return "asr";
  }
  if (/tts|text-to-speech|vits|speecht5|bark|fastspeech|tacotron|vocoder/.test(haystack)) return "tts";
  return "llm";
}

export function getConfigNumber(config: Record<string, unknown> | null | undefined, keys: string[]): number | undefined {
  if (!config) return undefined;
  for (const key of keys) {
    const value = config[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) return Number(value);
  }
  return undefined;
}

export function estimateLlmParams(config: Record<string, unknown> | null | undefined): number | undefined {
  const hidden = getConfigNumber(config, ["hidden_size", "n_embd", "d_model"]);
  const layers = getConfigNumber(config, ["num_hidden_layers", "n_layer", "num_layers"]);
  const heads = getConfigNumber(config, ["num_attention_heads", "n_head", "encoder_attention_heads"]);
  const kvHeads = getConfigNumber(config, ["num_key_value_heads", "num_kv_heads"]) || heads;
  const vocab = getConfigNumber(config, ["vocab_size"]);
  const intermediate = getConfigNumber(config, ["intermediate_size", "ffn_dim"]) || (hidden ? hidden * 4 : undefined);

  if (!hidden || !layers || !heads || !kvHeads || !vocab || !intermediate) return undefined;

  const headDim = getConfigNumber(config, ["head_dim"]) || hidden / heads;
  const qProj = hidden * heads * headDim;
  const kvProj = 2 * hidden * kvHeads * headDim;
  const outProj = hidden * hidden;
  const gatedMlp = 3 * hidden * intermediate;
  const norms = 4 * hidden;
  const perLayer = qProj + kvProj + outProj + gatedMlp + norms;
  const embeddings = vocab * hidden;
  const tied = config?.tie_word_embeddings !== false;
  return embeddings + layers * perLayer + (tied ? 0 : embeddings);
}

export function estimateParamsFromMetadata(metadata: ModelMetadata | null, kind: ModelKind): number | undefined {
  const config = metadata?.config || undefined;
  const explicit = getConfigNumber(config, ["num_parameters", "n_params", "parameter_count"]);
  if (explicit) return explicit;

  const raw = metadata?.raw || {};
  const safetensors = raw.safetensors as { total?: number; parameters?: Record<string, number> } | undefined;
  if (safetensors?.parameters) {
    const total = Object.values(safetensors.parameters).reduce((sum, value) => sum + Number(value || 0), 0);
    if (total > 0) return total;
  }

  if (kind === "llm") {
    const llmParams = estimateLlmParams(config);
    if (llmParams) return llmParams;
  }

  if (metadata?.storageBytes) {
    return metadata.storageBytes / 2;
  }

  return undefined;
}
