import type { RequirementReport } from "./types";

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 GB";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1000 && unitIndex < units.length - 1) {
    value /= 1000;
    unitIndex += 1;
  }
  return `${formatNumber(value)} ${units[unitIndex]}`;
}

export function formatGib(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 GiB";
  if (bytes < 1024 ** 2) return `${formatNumber(bytes)} B`;
  if (bytes < 1024 ** 3) return `${formatNumber(bytes / 1024 ** 2)} MiB`;
  return `${formatNumber(bytes / 1024 ** 3)} GiB`;
}

export function formatNumber(value: number, maximumFractionDigits = 2): string {
  if (!Number.isFinite(value)) return "-";
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits,
    minimumFractionDigits: value >= 10 ? 0 : 1
  }).format(value);
}

export function formatComputeTops(value: number): string {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs > 0 && abs < 0.01) return `${formatNumber(value * 1000)} GOPS`;
  return `${formatNumber(value)} TOPS`;
}

export function formatComputeOps(ops: number): string {
  if (!Number.isFinite(ops)) return "-";
  const abs = Math.abs(ops);
  if (abs >= 1e12) return `${formatNumber(ops / 1e12)} TOp`;
  if (abs >= 1e9) return `${formatNumber(ops / 1e9)} GOp`;
  if (abs >= 1e6) return `${formatNumber(ops / 1e6)} MOp`;
  return `${formatNumber(ops)} Op`;
}

export function formatBand(value: number, unit: string): string {
  if (unit === "GB") {
    if (Math.abs(value) > 0 && Math.abs(value) < 0.01) return `${formatNumber(value * 1000)} MB`;
    return `${formatNumber(value)} GB`;
  }
  if (unit === "GB/s") return `${formatNumber(value)} ${unit}`;
  if (unit === "TOPS" || unit === "TFLOPs") return formatComputeTops(value);
  if (unit === "Op") return formatComputeOps(value);
  if (unit === "MB") return `${formatNumber(value)} MB`;
  if (unit === "ms") return `${formatNumber(value)} ms`;
  if (unit === "tokens/s" || unit === "FPS" || unit === "RTF") return `${formatNumber(value)} ${unit}`;
  return `${formatNumber(value)} ${unit}`;
}

export function confidenceLabel(confidence: RequirementReport["confidence"]): string {
  if (confidence === "high") return "高";
  if (confidence === "medium") return "中";
  return "低";
}

export function reportToMarkdown(report: RequirementReport): string {
  const metricRows = report.metrics
    .map(
      (metric) =>
        `| ${metric.label} | ${formatBand(metric.value.aggressive, metric.unit)} | ${formatBand(
          metric.value.typical,
          metric.unit
        )} | ${formatBand(metric.value.conservative, metric.unit)} | ${metric.note || ""} |`
    )
    .join("\n");
  const keyFormulas = report.formulas.filter((row) => row.level === "key");
  const advancedFormulas = keyFormulas.length ? report.formulas.filter((row) => row.level !== "key") : [];
  const keyFormulaRows = (keyFormulas.length ? keyFormulas : report.formulas)
    .map((row) => `| ${row.item} | ${row.formula} | ${row.inputs} | ${row.result} |`)
    .join("\n");
  const advancedFormulaRows = advancedFormulas
    .map((row) => `| ${row.item} | ${row.formula} | ${row.inputs} | ${row.result} |`)
    .join("\n");
  const assumptions = report.assumptions.map((item) => `- ${item}`).join("\n");
  const warnings = report.warnings.map((item) => `- ${item}`).join("\n") || "- 无";

  return `# ${report.title}

## 结论
${report.summary}

置信度：${confidenceLabel(report.confidence)}

## 指标
| 指标 | 激进 | 典型 | 保守 | 说明 |
|---|---:|---:|---:|---|
${metricRows}

## 关键公式
| 项目 | 公式 | 输入 | 结果 |
|---|---|---|---|
${keyFormulaRows}

${advancedFormulaRows ? `## 高级公式\n| 项目 | 公式 | 输入 | 结果 |\n|---|---|---|---|\n${advancedFormulaRows}\n` : ""}

## 假设
${assumptions}

## 风险
${warnings}
`;
}
