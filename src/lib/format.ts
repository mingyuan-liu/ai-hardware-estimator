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
  return `${formatNumber(bytes / 1024 ** 3)} GiB`;
}

export function formatNumber(value: number, maximumFractionDigits = 2): string {
  if (!Number.isFinite(value)) return "-";
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits,
    minimumFractionDigits: value >= 10 ? 0 : 1
  }).format(value);
}

export function formatBand(value: number, unit: string): string {
  if (unit === "GB" || unit === "GB/s") return `${formatNumber(value)} ${unit}`;
  if (unit === "TOPS" || unit === "TFLOPs") return `${formatNumber(value)} ${unit}`;
  if (unit === "MB") return `${formatNumber(value)} MB`;
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
  const formulas = report.formulas
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

## 公式
| 项目 | 公式 | 输入 | 结果 |
|---|---|---|---|
${formulas}

## 假设
${assumptions}

## 风险
${warnings}
`;
}
