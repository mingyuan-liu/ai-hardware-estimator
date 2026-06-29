import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number(process.env.PORT || 8787);
const HUGGINGFACE_BASE_URL = (process.env.HUGGINGFACE_BASE_URL || "https://huggingface.co").replace(/\/$/, "");

app.use(express.json({ limit: "1mb" }));

function encodeModelId(modelId) {
  return modelId
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}${body ? `: ${body.slice(0, 180)}` : ""}`);
  }
  return response.json();
}

async function fetchTextIfAvailable(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) return null;
  return response.text();
}

function safeJsonParse(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function normalizeFile(file) {
  return {
    name: file.rfilename || file.path || file.name || "",
    size: Number(file.size || file.Size || 0) || undefined,
    lfs: file.lfs ? { size: Number(file.lfs.size || 0) || undefined } : undefined
  };
}

function sumKnownStorage(files, fallback) {
  const total = files.reduce((sum, file) => {
    const size = file.size || file.lfs?.size || 0;
    return sum + size;
  }, 0);
  return total || fallback || undefined;
}

function dtypeBytes(config) {
  const dtype = String(config?.torch_dtype || config?.dtype || "").toLowerCase();
  if (dtype.includes("int4")) return 0.5;
  if (dtype.includes("int8")) return 1;
  if (dtype.includes("float32") || dtype.includes("fp32")) return 4;
  return 2;
}

function extractModelScopeFiles(info) {
  const candidates = [
    info?.Data?.ModelInfos?.safetensor?.files,
    info?.Data?.ModelInfos?.pytorch?.files,
    info?.Data?.ModelInfos?.onnx?.files,
    info?.Data?.Files
  ];
  const files = candidates.find((value) => Array.isArray(value)) || [];
  return files.map(normalizeFile).filter((file) => file.name);
}

function normalizeModelScopeConfig(configText) {
  const parsed = safeJsonParse(configText);
  if (!parsed || !parsed.Code) return parsed;
  return safeJsonParse(parsed.Data) || parsed.Data || parsed;
}

async function getHuggingFaceModel(modelId) {
  const encoded = encodeModelId(modelId);
  const info = await fetchJson(`${HUGGINGFACE_BASE_URL}/api/models/${encoded}`);
  const config = safeJsonParse(await fetchTextIfAvailable(`${HUGGINGFACE_BASE_URL}/${encoded}/raw/main/config.json`));
  const files = (info.siblings || []).map(normalizeFile).filter((file) => file.name);
  const safetensorsStorage =
    info.safetensors?.total && Number(info.safetensors.total) > 0
      ? Number(info.safetensors.total) * dtypeBytes(config)
      : undefined;
  const storageBytes = sumKnownStorage(files, safetensorsStorage || info.usedStorage);

  return {
    source: "huggingface",
    modelId,
    name: info.modelId || modelId,
    pipelineTag: info.pipeline_tag,
    tags: info.tags || [],
    config,
    files,
    storageBytes,
    raw: {
      downloads: info.downloads,
      likes: info.likes,
      libraryName: info.library_name,
      createdAt: info.createdAt,
      lastModified: info.lastModified,
      safetensors: info.safetensors
    }
  };
}

async function getModelScopeModel(modelId) {
  const encoded = encodeModelId(modelId);
  const info = await fetchJson(`https://modelscope.cn/api/v1/models/${encoded}`);
  const configText = await fetchTextIfAvailable(
    `https://modelscope.cn/api/v1/models/${encoded}/repo?Revision=master&FilePath=config.json`
  );
  const config = normalizeModelScopeConfig(configText);
  const files = extractModelScopeFiles(info);
  const storageBytes = sumKnownStorage(files, Number(info?.Data?.StorageSize || 0) || undefined);
  const data = info.Data || {};

  return {
    source: "modelscope",
    modelId,
    name: data.Name || modelId,
    pipelineTag: data.Tasks?.[0]?.Name,
    tags: [...(data.ModelType || []), ...(data.Frameworks || []), ...(data.Libraries || [])].filter(Boolean),
    config,
    files,
    storageBytes,
    raw: {
      downloads: data.Downloads,
      stars: data.Stars,
      license: data.License,
      architectures: data.Architectures,
      modelInfos: data.ModelInfos,
      lastUpdatedTime: data.LastUpdatedTime
    }
  };
}

app.get("/api/model", async (req, res) => {
  const source = String(req.query.source || "");
  const modelId = String(req.query.modelId || "").trim();

  if (!modelId) {
    res.status(400).json({ error: "modelId is required" });
    return;
  }

  try {
    if (source === "huggingface") {
      res.json(await getHuggingFaceModel(modelId));
      return;
    }
    if (source === "modelscope") {
      res.json(await getModelScopeModel(modelId));
      return;
    }
    res.status(400).json({ error: "source must be huggingface or modelscope" });
  } catch (error) {
    res.status(502).json({
      error: error instanceof Error ? error.message : "Failed to fetch model metadata"
    });
  }
});

app.get("/api/health", (_req, res) => {
  res.json({ ok: true });
});

const distDir = path.resolve(__dirname, "../dist");
app.use(express.static(distDir));
app.get("*", (_req, res) => {
  res.sendFile(path.join(distDir, "index.html"));
});

app.listen(PORT, HOST, () => {
  console.log(`AI hardware requirement web app listening on http://${HOST}:${PORT}`);
});
