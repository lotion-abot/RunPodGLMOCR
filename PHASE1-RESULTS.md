# Phase 1 结果 — GREEN(2026-08-11,RunPod Pod `7flxvv3va0w9aj`,NVIDIA L4 24GB,EU-RO-1)

自建 GLM-OCR 在真实审计页面上跑通,并量出了第一个**作数**的并发数字。

---

## 1. 三个谜团全破

### ① `md=0` 的真凶 = layout 检测 CUDA OOM,**不是版本漂移**

pod 的 `glmocr_5004.log` 实证:

```
Layout detection failed for pages [0], skipping batch: CUDA out of memory.
Tried to allocate 276.00 MiB. GPU 0 has 22.03 GiB of which 255.69 MiB is free.
Process 5844 (vLLM) has 16.69 GiB. Process 6664 has 1.80 GiB. Process 6714 has 1.80 GiB.
```

OOM 之后 glmocr **跳过该批次但照样返回 HTTP 200 + 空 markdown** —— 快、静默、和成功
无法区分。这就是 Replicate T4(16GB,更挤)上每页都空的原因。

**显存账**:vLLM 吃 `gpu-memory-utilization × VRAM`,**每个 glmocr 进程的 layout 模型
额外还要 ~1.9 GiB**。

| util | vLLM | 3×layout | 合计 / 22.03 GiB |
|---|---|---|---|
| 0.70(原) | 16.69 GiB | 5.4 GiB | **22.1 → 炸** |
| **0.55(现)** | 13.49 GiB | 5.78 GiB | 19.3 → 稳 |

修完:**6/6 全过,三个日志 OOM 计数全 0**。

### ② bbox 坐标系:自建是 0-1000 归一化,云端是像素

源码实证:
- `glmocr/layout/base.py:42` → self-hosted `bbox_2d` = **normalized (0-1000)**
- `glmocr/api.py:353` → MaaS(云)= **absolute pixel coordinates**

换算公式 `x_px = round(x*W/1000)`,拿 Directors' Report 那一页跟 Z.ai 黄金样本**逐元素对**:

| 元素 | Z.ai 黄金 | 我们(换算后) | 差 |
|---|---|---|---|
| `## Directors' Report` | `[538,505,1000,577]` | `[535,504,997,573]` | 3px |
| 正文段 | `[533,639,2857,772]` | `[532,639,2856,770]` | 1px |
| `## Principal Activities` | `[537,904,1037,974]` | `[535,901,1035,971]` | 3px |
| Financial Results 表 | `[533,1387,2861,1729]` | `[532,1384,2859,1724]` | 3px |
| `## CHEN XIAOQING...` | `[536,2913,983,3047]` | `[535,2912,981,3043]` | 2px |

**公式确认正确**,1-4px 的差纯粹是 0-1000 量化的舍入。

⚠️ 没有这个换算,`ReportStitchOcr` 的缝合、`HeaderRecovery`、`BboxAspect` 会**静默错位** ——
所有字段看起来都正常。

### ③ 6-7 分钟冷启动里,**模型加载只占 0.7 秒**

```
Loading weights took 0.43 seconds
Loading weights took 0.27 seconds
init engine (profile, create kv cache, warmup model) took 301.12 seconds   <-- 全在这
```

**把权重烘进镜像救不了冷启动。** 要动的是 CUDA graph / torch.compile 预热:
缓存 compile 产物,或 `--enforce-eager`(牺牲推理速度),或砍 `cudagraph_capture_sizes`。

---

## 2. 正确性 — 通过

| 检查 | 结果 |
|---|---|
| 真实 3166×4096 页出 markdown | ✅ 628-2082 字符,表格识别为 `<table border="1">` |
| `native_label` 词表 | ✅ `paragraph_title` / `table` / `text` —— 与云端一致 |
| 元素字段 | ✅ `bbox_2d` `content` `index` `label` `native_label` 齐全 |
| bbox 空间 | ⚠️ 0-1000 归一化,**适配器必须换算**(公式已验证) |
| `layout_details` 结构 | 嵌套 `[[...]]`,与云端一致 |

**适配器要补的小差异**(都在 Phase 2 的 handler 里做,已有现成代码):
- 删掉多出来的 `polygon`
- 补上每元素的 `height`/`width`(云端有,自建没有)
- `data_info.pages` 自建是 `[]` → 用输入图尺寸填
- `usage` 自建是 `{}` → 补 0(自建不按 token 计费,无所谓)
- 目录页会出现 `content` 这个 label(正文页不会)

---

## 3. 并发 — 第一个作数的数字

```
solo baseline: 1.31s/page      backends: 3 glmocr 进程 / 1 GPU

  N     wall   pages/s     avg     p95  parallel  ok
  1     1.8s      0.54    1.8s    1.8s     0.71x  1/1
  2     3.6s      0.56    3.1s    2.6s     0.73x  2/2
  3     4.6s      0.65    3.8s    3.4s     0.86x  3/3
  6     6.6s      0.91    5.6s    6.4s     1.19x  6/6
```

**峰值 0.91 pages/s** —— 6 页 6.6 秒。

对比基准要说清楚:Z.ai 那个「6 页 22 秒 = 0.27 pages/s」是**从马来西亚打到 Z.ai 云端的
端到端时间**(含网络往返和他们的排队);我们这 0.91 是**在 pod 内 localhost 量的纯 GPU 吞吐**。
所以这是 GPU 侧的对比,**RunPod 的真实端到端(马来西亚→EU-RO-1)还没测**。

> 之前那个「并行度 ~1.3」是 `md=0` 那轮跑出来的,测的是管线空转,**已作废**。

### ⚠️ 多进程架构可以整块删掉

同一个热 stack 上直接对比 1 / 2 / 3 个 glmocr 进程:

| backends | 峰值 pages/s | 并行度 | 显存 |
|---|---|---|---|
| **1** | **0.92** | 1.15x | ~2.0 GiB |
| 2 | 0.95 | 1.22x | ~4.0 GiB |
| 3 | 0.96 | 1.22x | ~5.8 GiB |

**多开 2 个进程只买到 4%,却多吃 3.8 GiB 显存。**

而且前提本身就是错的:`predict.py` 里写「glmocr 的 Flask `app.run()` 是单线程,一个进程
一次只能 parse 一页」—— **Flask 从 1.0 起 `app.run()` 默认 `threaded=True`**。一个进程
本来就能并发。整套「三进程 + 端口池」是为了解决一个不存在的问题。

**Phase 2 直接降到 1 个进程**:省 3.8 GiB → vLLM util 可以从 0.55 提回 0.70(更多 KV cache),
handler 里的端口池 / 借还逻辑全删,**顺便把当初造成 `md=0` 的 OOM 风险从根上拿掉**。

---

## 3b. 测试台 `runpod_test.py`(ladder + stress 合一)

```powershell
# 在 pod 上(纯 GPU 吞吐)
ENDPOINTS=http://localhost:5002 IMAGE_DIR=/workspace/pages MODE=both /opt/venv/bin/python runpod_test.py

# 从 Windows 打(GPU + 马来西亚→罗马尼亚网络)—— 先开隧道
ssh -N -L 5002:localhost:5002 -p 11053 -i $env:USERPROFILE\.ssh\id_ed25519 root@213.173.105.14
python runpod_test.py
```

`MODE` = `ladder` | `stress` | `both`。

### Ladder 的新用途:它是**找 OOM 悬崖的工具**

因为空 markdown 就是 layout 阶段 CUDA OOM 的签名,**第一次出现空的那一级 = 显存悬崖**,
所以这个 ladder 直接用来标定 `--gpu-memory-utilization`。实测(util=0.55,单进程):

| N | wall | pages/s(只算成功) | 空/失败 |
|---|---|---|---|
| 1 | 1.3s | 0.79 | 0 |
| 3 | 3.3s | 0.90 | 0 |
| 6 | 6.4s | **0.94** | 0 |
| 8 | 9.8s | 0.82 | 0 |
| **12** | 9.0s | 0.50 | **6/12** ← 悬崖 |

**诊断可证明正确**:N=12 那一轮之后 `grep -c 'out of memory' glmocr_5002.log` = **正好 6 条**,
与 6 个空结果一一对应。失败调用 1.9 秒就返回(OOM 跳过批次的「快速空」特征)。

**运行安全线:并发 ≤ 8。**

### Stress(30 调用 @ 6 并发,持续负载)

```
calls        : 30   ok: 30   FAIL/EMPTY: 0
burst wall   : 37.4s   throughput: 0.80 pages/s  (48 pages/min)
latency  p50 : 6.1s   p95: 12.1s   max: 12.2s   mean: 7.0s
```

持续 0.80 < 突发 0.94,正常。**28 页的报告 ≈ 35 秒 GPU 时间。**

---

## 4. 钉死的版本(Phase 2 Dockerfile 照抄)

```
vllm              0.19.1
glmocr            0.1.5
transformers      5.15.0
torch             2.10.0
torchvision       0.25.0
flashinfer-python 0.6.6
python            3.12.3
```

vLLM 启动参数(实测被接受,dtype 自动选 bfloat16):

```
--model zai-org/GLM-OCR --served-model-name default glm-ocr
--max-model-len 32768 --gpu-memory-utilization 0.55
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

MTP 推测解码**确认可用**(`speculative_config=SpeculativeConfig(method='mtp', num_spec_tokens=1)`)——
Replicate 那版从来没开过。

---

## 5. RunPod 运维实证(踩过的坑)

- **`ssh.runpod.io` 代理不可用于自动化**:不支持 exec、不支持 scp、只给 PTY,**且随机丢输入行**。
  必须用 pod 的**公网 IP + TCP 端口直连**(`RUNPOD_PUBLIC_IP` / `RUNPOD_TCP_PORT_22`)。
- **账号里的 SSH key 只在 pod 启动时注入**。pod 起来之后加的 key,要手动 append 进
  `/root/.ssh/authorized_keys` 才能直连。
- **`/workspace` 是网络文件系统**(`mfs#euro.runpod.net`),不是本地盘。venv 放 `/opt/venv`
  (本地 20GB 容器盘,用了 ~13GB),只把模型放 `/workspace`。
- 分配到的是 6 vCPU / 62 GB RAM / L4 24GB。

---

## 6. 一个还没定论的行为差异

同一页(Directors' Report),**我们 13 个元素,Z.ai 黄金样本 12 个**。多出来的是 index 4
`## Financial Results`(`paragraph_title`),它的 bbox `[532,1384,2859,1724]` 跟 index 5
那张表的 bbox **重叠** —— Z.ai 把这个标题折进表里了,我们单独吐出来。

`LayoutTitleDeduper` 按 `native_label` 判定,可能折得掉也可能折不掉。**Phase 2 要验**,
别等以后当成灵异事件重新查一遍。

---

## 7. Phase 2 待办

1. **降到 1 个 glmocr 进程**(见 §3),vLLM util 提回 0.70,删掉端口池
2. Dockerfile:钉死 §4 的版本,烘焙两个模型
3. handler:**bbox 换算**(§1②)+ polygon/height/width/data_info/usage 补齐 + 空 markdown 毒丸重试
4. 攻冷启动的 301 秒(compile cache / `--enforce-eager` / 砍 `cudagraph_capture_sizes`)
5. 量**真实端到端延迟**(马来西亚 → RunPod),这才是用户感知的数字
6. 验 §6 那个多出来的标题元素
7. 全绿后才动 C#(`GlmOcrClient.cs` 加 RunPod provider)
