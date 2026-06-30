# AI 硬件需求评估工具

这是一个面向 AI 推理部署的硬件需求估算 Web 工具。它可以根据模型类型、推理负载、量化精度和部署目标，估算内存/显存容量、模型存储、算力、内存带宽以及潜在瓶颈风险。

当前覆盖：

- LLM
- ASR
- TTS
- YOLO / CV

## 功能特性

- 内置模型列表，按模型类型、厂商/组织、模型名称选择。
- 支持从 Hugging Face 和 ModelScope 获取公开模型元数据。
- 内置 YOLOv8、YOLO11 的检测、分割、姿态、OBB、分类任务估算。
- LLM 支持按 benchmark 习惯输入 `PP token/s` 和 `TG token/s`。
- LLM 报告区分 Prefill 和 Decode。
- Decode 带宽拆分为权重带宽、KV 带宽和总带宽。
- YOLO 报告区分关键公式和高级公式。
- 支持导出 JSON 和 Markdown 报告。
- 页面底部提供 GOp、TOp、GOPS、TOPS、GB、GiB、GB/s 等单位速查。

## 支持的工作负载

### LLM

LLM 输入默认采用 benchmark 风格：

- `Benchmark 长度`：prompt 和生成 token 的长度。
- `PP token/s`：Prefill 阶段吞吐。
- `TG token/s`：Decode / 生成阶段吞吐。
- `Batch / 并发`：高级设置中可调整。
- `最大 Context tokens`：用于估算 KV cache。

LLM 报告会拆分：

- 权重容量
- KV cache
- Prefill 计算量
- Prefill TOPS
- Decode 权重带宽
- Decode KV 带宽
- Decode 总带宽
- Decode TOPS

### YOLO / CV

YOLO 输入包括：

- 模型规模和任务类型
- 输入尺寸
- Batch
- 单路 FPS
- 类别数
- 精度
- 后端有效利用率模式

关键公式默认展示：

- 总内存容量
- 主网络算力
- DDR/显存带宽
- 后处理工作区

高级公式中包含：

- 参数量
- 模型权重
- 模型存储
- 输入尺寸系数
- YOLO FLOPs
- 特征图峰值
- 候选数
- 输出字段数
- 输出 tensor
- 运行时余量
- 单 batch 带宽访问量

### ASR / TTS

ASR 和 TTS 报告会估算：

- 模型内存
- 运行时激活/缓存
- 模型存储
- 算力需求
- 内存带宽
- 实时率 RTF 相关需求

## 单位说明

- `GOp`：总操作量，1 GOp = 10 亿次操作。
- `TOp`：总操作量，1 TOp = 1 万亿次操作 = 1000 GOp。
- `GOPS`：每秒操作量，1 GOPS = 每秒 10 亿次操作。
- `TOPS`：每秒操作量，1 TOPS = 每秒 1 万亿次操作。
- `GB`：十进制容量，1 GB = 1,000,000,000 bytes。
- `GiB`：二进制容量，1 GiB = 1,073,741,824 bytes。
- `GB/s`：带宽，表示每秒搬运多少十进制 GB 数据。

简单记法：

```text
Op / GOp / TOp   = 一共要做多少操作
OPS / GOPS / TOPS = 每秒能做多少操作
GB / GiB         = 容量
GB/s             = 带宽
```

## 本地运行

安装依赖：

```bash
npm install
```

构建：

```bash
npm run build
```

启动完整应用和 API 代理：

```bash
npm start
```

浏览器打开：

```text
http://localhost:8787
```

前端开发模式：

```bash
npm run dev
```

开发模式下，Vite 会把 `/api` 请求代理到：

```text
http://localhost:8787
```

因此如果需要实时获取 Hugging Face / ModelScope 元数据，开发时仍需要同时运行：

```bash
npm start
```

## 部署说明

项目包含一个 Node/Express API 代理：

```text
server/index.mjs
```

这个代理用于：

- 获取 Hugging Face 模型元数据。
- 获取 ModelScope 模型元数据。
- 避免浏览器 CORS 问题。

因此，纯静态托管不能完整运行所有功能：

- GitHub 仓库页面只能展示源码。
- GitHub Pages 可以托管前端构建结果，但无法运行 `/api/model`。
- 如果使用 GitHub Pages，需要把 API 代理部署到其他服务，并让前端访问那个 API 地址。

完整部署建议使用可以运行 Node.js 的平台：

- Vercel
- Render
- Railway
- Fly.io
- 自有 VPS

### www.laumy.tech 子路径部署

当前生产部署路径：

```text
https://www.laumy.tech/tools/ai-hardware-estimator/
```

部署到服务器：

```bash
git clone git@github.com:mingyuan-liu/ai-hardware-estimator.git /srv/ai-hardware-estimator
cd /srv/ai-hardware-estimator
npm ci
npm run build
```

服务由 systemd 管理：

```bash
sudo cp deploy/systemd/ai-hardware-estimator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-hardware-estimator.service
```

服务只监听本机：

```text
HOST=127.0.0.1
PORT=8787
HUGGINGFACE_BASE_URL=https://hf-mirror.com
```

Nginx 通过 `/tools/ai-hardware-estimator/` 反向代理到本机 `8787` 端口。

### 自动部署

服务器使用独立 deploy service 从 GitHub 拉取并重建：

```bash
sudo cp deploy/systemd/ai-hardware-estimator-deploy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start ai-hardware-estimator-deploy.service
```

部署脚本：

```text
scripts/update_from_git.sh
```

默认配置：

```text
APP_DIR=/srv/ai-hardware-estimator
APP_BRANCH=main
APP_SERVICE=ai-hardware-estimator.service
```

GitHub webhook 复用主站的 `/deploy-hook`，由主站 webhook 服务按仓库名触发：

```text
mingyuan-liu/ai-hardware-estimator -> ai-hardware-estimator-deploy.service
```

## 精度和校准说明

本工具用于早期硬件规格评估和方案评审，不是周期级仿真器。

当前估算口径：

- LLM Decode 带宽由有效权重读取和 KV cache 读写估算。
- LLM Prefill 计算量由参数量、batch、prompt 长度和 attention 二次项估算。
- YOLO 激活、输出、后处理和带宽使用经验系数估算。
- ASR/TTS 使用模型规模和实时率 RTF 做粗估。

最终硬件决策前，建议使用真实 benchmark 或 profiler 数据进行校准。

## 项目结构

```text
server/
  index.mjs              # Express 服务和模型元数据代理

src/
  App.tsx                # 页面 UI 和报告渲染
  lib/catalog.ts         # 内置 YOLO 模型目录
  lib/estimators.ts      # LLM / ASR / TTS / YOLO 估算逻辑
  lib/format.ts          # 单位和报告格式化
  lib/model.ts           # 模型类型和参数量推断
  lib/types.ts           # 共享类型定义
```
