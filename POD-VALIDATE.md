# Phase 1 — 在 RunPod Pod 上把 GLM-OCR 跑绿(手动、可交互、便宜)

**目的:** 在冻成 serverless 之前,先用真实审计页面证明这套 stack 真的出 markdown,
并量出**真实的并发上限**,同时把每个包的确切版本抄下来。

Replicate 那边 `md=0` 的病因**还没定罪**(版本漂移只是嫌疑最大的)。Pod 阶段的价值
就是**有实时日志可以看**,而 serverless 只能事后捞。所以这一步不能跳。

Phase 2(Dockerfile + serverless endpoint)等这里全绿再做。

---

# 1. 开 Pod — 逐项照抄

到 [runpod.io](https://www.runpod.io) → 左边 **Pods** → **Deploy** / **+ Deploy a Pod**。

## 1.1 选 GPU

| 项目 | 选什么 | 为什么 |
|---|---|---|
| **GPU** | **RTX A4000**(16GB)。没货就 **RTX A4500 / RTX 4000 Ada**,同样 16GB | GLM-OCR 只有 0.9B,16GB 绰绰有余。而且 serverless 最便宜的一档就是 16GB —— **现在就在同一档上验,量出来的数字才作数** |
| **Cloud Type** | **Community Cloud**(验证阶段) | ~$0.17/hr。⚠️ 将来跑真实客户审计文件要换 **Secure Cloud**(RunPod 自己的机房),Community 是**别人的机器** |
| **CUDA 筛选** | 如果有 CUDA version 筛选器,选 **12.8 以上** | venv 里 pip 装的 torch 自带 CUDA runtime,但**宿主机的 NVIDIA 驱动**得够新 |
| 数量 | 1 | |

## 1.2 选 Template ← **你问的就是这个**

模板列表里搜 **`PyTorch`**,选官方的 **RunPod PyTorch**(蓝色 RunPod 官方标)。

然后点 **Edit Template**,把 **Container Image** 那栏**改成这个确切的 tag**:

```
runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404
```

> **为什么是这一个:**
> - `1.0.7` = 目前最新的**稳定**版(2026-07-23),不是 `1.1.0-rc.x` 那种 release candidate
> - `cu1281` = CUDA 12.8.1,够新
> - `ubuntu2404` = Ubuntu 24.04 底,系统 Python **大概率**是 3.12(我按 base image 推的,
>   没进容器确认过)—— 所以脚本才要自己探测,见下
> - RunPod 官方 PyTorch 模板**自带 JupyterLab(8888)+ SSH**,拖文件进去最省事
>
> **模板里的 torch 版本无所谓** —— 我们的脚本自己建一个隔离 venv,vllm 会装它自己
> 要的 torch。选这个模板纯粹是为了 **CUDA 驱动 + Jupyter + SSH** 这三样现成的。
>
> 脚本会**自动探测** Python 版本(3.10/3.11/3.12 都收),所以万一你换了别的模板也不会
> 直接炸 —— 它会打印用的是哪个解释器,不合格就当场报错让你换模板。

## 1.3 Edit Template 里其余四栏

还在 **Edit Template** 面板里:

| 栏位 | 填什么 | 为什么 |
|---|---|---|
| **Container Disk** | `20` GB(默认就行) | 系统盘,pod 删了就没了 |
| **Volume Disk** | **`60`** GB ← **一定要改** | venv(~8GB)+ 两个模型都放这。**这块盘 Stop 了还在**,重开 pod 不用重装重下 |
| **Volume Mount Path** | `/workspace`(默认) | 脚本里所有路径都按这个写死 |
| **Expose HTTP Ports** | `8888,5002,5003,5004` ← **一定要改** | `8888` 是 Jupyter(默认已有,别删);`5002-5004` 是三个 glmocr 进程 |
| Expose TCP Ports | `22`(默认,留着) | SSH |

可选:**Environment Variables** 加一条 `JUPYTER_PASSWORD` = 你自己设个密码。不设的话密码在 pod 日志里。

## 1.4 Deploy

**Set Overrides / Save** → 回到部署页 → **On-Demand**(不要 Spot,Spot 会被抢占,跑一半没了)→ **Deploy**。

等状态变 **Running**(1-3 分钟,拉镜像)。

> 💰 **Pod 是按小时一直计费的,不会像 serverless 那样归零。** 验完记得 **Stop**。
> Stop 之后只收 Volume 的钱(60GB 大概 $0.0042/hr,一个月几块钱),下次 Start 秒回。

---

# 2. 把两个文件放进 pod

Pod 卡片上点 **Connect** → **Jupyter Lab [Port 8888]**(浏览器开一个新页)。

左边文件树默认就在 `/workspace`。把这两个文件**直接从 Windows 资源管理器拖进去**:

```
C:\RunPodGLMOcr\pod_setup.sh
C:\RunPodGLMOcr\runner.py
```

拖完左边应该看到 `pod_setup.sh` 和 `runner.py` 两个文件。

<details>
<summary>拖拽不行的备用方案</summary>

pod 上预装了 `runpodctl`。Windows 这边装一个([runpodctl releases](https://github.com/runpod/runpodctl/releases) 下 `runpodctl-windows-amd64.exe`),然后:

```powershell
cd C:\RunPodGLMOcr
runpodctl send pod_setup.sh      # 会吐一个一次性码,例如 8338-galileo-collect-fidel
```

pod 的终端里:
```bash
cd /workspace && runpodctl receive 8338-galileo-collect-fidel
```
两个文件各来一次。
</details>

---

# 3. 跑起来

Pod 卡片 → **Connect** → **Web Terminal**(或者 Jupyter 里 File → New → Terminal,一样的)。

先确认一下环境:

```bash
nvidia-smi                    # 看到 GPU 型号和显存 = 驱动正常
python3 -V                    # 打印出来看一眼,3.10 / 3.11 / 3.12 都行
ls -la /workspace             # 应该看到刚拖进去的两个文件
```

然后:

```bash
sed -i 's/\r$//' /workspace/pod_setup.sh /workspace/runner.py   # 防 Windows 换行符
bash /workspace/pod_setup.sh
```

> 第一行是保险:文件从 Windows 过来,万一带了 CRLF,bash 会报一句
> `$'\r': command not found` —— 报错完全看不出原因。文件本来就是 LF 的话这行也无害。

- **第一次约 10-15 分钟**:装 vllm+torch(~5GB)+ 下两个模型
- **之后重跑约 2 分钟**:venv 和模型都在 `/workspace`,跳过

跑的过程会打印 6 个阶段。看到这个就是全起来了:

```
=== 6/6  READY ===
  vLLM     : http://localhost:8080
  glmocr   : http://localhost:5002/glmocr/parse
  glmocr   : http://localhost:5003/glmocr/parse
  glmocr   : http://localhost:5004/glmocr/parse
```

中间第 2 步会打印这一行,**截图给我**:

```
--- installed versions (RECORD THESE) ---
vllm         0.19.1
glmocr       0.1.5
transformers 5.x.x
torch        2.x.x
```

## 卡住 / 报错就看日志

```bash
tail -f /workspace/logs/vllm.log            # vLLM 起不来看这个
tail -f /workspace/logs/glmocr_5002.log     # OCR 结果空看这个
```

### 已知会踩的坑(都已在脚本里处理,列出来是为了让你看懂日志)

| 症状 | 原因 | 脚本里的对策 |
|---|---|---|
| `The model 'glm-ocr' does not exist` | glmocr 用包配置时发 `glm-ocr`,用最小配置时发 `default` | `--served-model-name default glm-ocr` 两个都服务 |
| `maximum context length is 8192 ... requested 8192 output tokens` | glmocr 要 8192 个**输出** token,上下文得留更多 | `--max-model-len 32768` |
| 所有 `native_label` 都是 `text` | 手写最小配置丢了 `label_task_mapping` | `runner.py` 拿包默认配置做 base 深合并;合并失败**直接报错**,不再静默降级 |
| HTTP 200 但 markdown 空 | **原因未定** —— Replicate 那边已经预烘焙过 layout 模型还是照样空,所以「模型没下全」不是已证实的病因 | 脚本先把两个模型都预下(排除这一种),真病因看 `vllm.log` + `glmocr_5002.log` |
| `--speculative-config` 被 vLLM 拒绝 | 这个版本不认 MTP 的写法 | 关掉重跑:`USE_MTP=0 bash /workspace/pod_setup.sh` |
| `NO USABLE PYTHON` | 模板的 Python 不在 3.10-3.12 | 换回 1.2 里那个确切 tag |

---

# 4. 从 Windows 打真实页面

Pod 页面上复制 **Pod ID**(卡片标题下面那串,像 `abc123xyz456`)。

填进 `C:\RunPodGLMOcr\pod_test.py` 顶部:

```python
POD_ID = "abc123xyz456"        # <- 改这里
```

然后 Windows 跑:

```powershell
cd C:\RunPodGLMOcr
python pod_test.py
```

它做三件事:

**预热** —— 三个 backend 各打一次。每个 glmocr 进程**第一次**请求才加载自己那份 layout
模型;不预热的话这个一次性开销会算进并发墙钟,又量出一个假的低并行度。

**A. 正确性** —— 一张真实的 3166×4096 审计页:
- `md_results` 非空 ← 这就是 Replicate 挂掉的地方
- `layout_details` 元素数 + 四个字段(`bbox_2d` / `native_label` / `content` / `index`)齐全
- `native_label` 词表不能只有 `['text']`
- **bbox 是像素还是归一化坐标** ← 如果是归一化的,字段全对、markdown 也有,但缝合信头会静默错位
- 结果存 `C:\RunPodGLMOcr\pod_output.json`

**B. 并发阶梯** —— 1 / 2 / 3 / 6 并发,输出真实的 pages/s 和有效并行度。

> 之前那个「并行度 ~1.3」是 `md=0` 那轮跑出来的,测的是管线空转,**作废**。
> 这次是第一个作数的数字。

---

# 5. 绿卡长什么样

```
md length         : 3000+
layout_details    : 20+ elements
element fields    : OK
native_label vocab: ['paragraph_title', 'table', 'text']     <- 不能只有 ['text']
bbox max x/y      : 3100/4050   (image is 3166x4096)         <- 不能是 <=1000
PEAK THROUGHPUT   : x.xx pages/s at N=?
```

五条都对 = Python 侧绿卡 → 才进 Phase 2。
这跟你定的规矩一致:**Python 这边绿了,才动 C#。**

---

# 6. 顺手在 pod 上记下来(Phase 2 要用)

**① 确切版本号**(第 2 阶段那几行,或者随时 `/workspace/venv/bin/pip list | grep -Ei "vllm|glmocr|transformers|torch"`)
→ Phase 2 的 Dockerfile 要一模一样钉死。**这是这次迁移最值钱的一件事。**

**② 6 并发时的 GPU 占用** —— 另开一个终端,趁 `pod_test.py` 跑的时候:
```bash
watch -n 1 nvidia-smi
```
看 GPU-Util 和 Memory-Usage → 决定 serverless 选 16GB 还是 24GB 档。

**③ 模型加载花了多久** —— `head -50 /workspace/logs/vllm.log` 看时间戳。
Phase 2 要把模型**烘进镜像**,把冷启动从 6-7 分钟压下来 —— 这比任何并发参数都值钱,
而且它才让 scale-to-zero 真正可用。

---

# 7. 验完关掉

Pod 卡片 → **Stop**。不 Stop 就一直按小时烧钱。

下次继续:**Start** → `bash /workspace/pod_setup.sh`(2 分钟,venv 和模型都还在)。
