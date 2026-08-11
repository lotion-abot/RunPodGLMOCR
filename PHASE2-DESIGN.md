# Phase 2 设计 — RunPod Serverless handler

前置:Phase 1 已绿(见 `PHASE1-RESULTS.md`)。本文的每一条都基于 pod 上的实测,不是推断。

---

## 1. 先删(Musk 第 2 步)

| 删掉 | 证据 |
|---|---|
| **三进程 + 端口池** | 实测 1/2/3 进程 = 0.92/0.95/0.96 pages/s。多开只买 **4%**,多吃 3.8 GiB 显存。而且前提是错的:`predict.py` 写「Flask `app.run()` 单线程」,**Flask 1.0 起默认 `threaded=True`** |
| **cog + 隔离 venv** | cog 钉 pydantic v1、vllm/glmocr 要 v2,整套 `/opt/vllm-venv` 是为绕开 cog 存在的。RunPod 用普通 Dockerfile,一个环境装到底 |
| **合成图自检** | 400×120 的假图能过,而真实 3166×4096 页全空 —— 这个自检**零价值**。换成镜像里烘一张真页 |
| **保温 PATCH 逻辑**(`EnsureWarmAsync` / check-then-patch) | Replicate deployment 专用。RunPod 有原生 idle timeout,整块删 |
| **启动时生成配置** | `runner.py` 的合并结果在**构建时**就固化进镜像,启动时不再跑 |

`runner.py` 的**合并逻辑本身保留** —— 丢了 `label_task_mapping` 所有 `native_label` 会退化成 `text`,C# 的标题识别 / 去重 / 页眉恢复会集体失明。只是把它从「每次启动跑」挪到「构建一次」。

---

## 2. 端点类型:**Queue-based,不是 Load-balancing**

LB 端点文档写死 **`Request timeout: 2 min (no worker available)`**,我们的冷启动是 **6-7 分钟** ——
每一次冷启动都必然超时失败。Queue 端点没有这个限制,请求在队列里等。

等冷启动压到 2 分钟以内,再重新评估 LB(它能省掉一层排队,延迟更低)。

---

## 2b. Pod(现在)vs Serverless(Phase 2)—— 决策记录

| | **Pod(现在跑的)** | **Serverless(Phase 2)** |
|---|---|---|
| 计费 | 按小时,**Running 就一直烧**,不归零 | 按秒,**空闲归零** |
| 冷启动 | 只有手动 Start 时;之后一直热 | 每次归零后 **6-7 分钟** |
| 扩展 | **一张卡封顶 0.94 pages/s**,加不了 | 自动加 worker;3 个 worker ≈ 2.8 pages/s |
| 排队 | **没有**,并发超了直接 OOM 出空 | 平台队列 + 自动重试 |
| 接口 | `POST http://host:5002/glmocr/parse` 同步返回 | `POST /v2/{id}/run` → 轮询 `/status/{jobid}` |
| 状态 | `/workspace` 持久,SSH 上去随便改 | **无状态**,改动必须重建镜像 |
| 部署 | scp 一个文件就生效 | build → push registry → 端点切版本 |
| 适合 | **开发调试**(我们现在正在做的事) | **生产**(突发、低频、要弹性) |

### 成本实算(24GB L4/A5000 档)

| 方案 | 单价 | 月成本 |
|---|---|---|
| Pod 常开(Community) | ~$0.39/hr | **$285** |
| Pod 常开(Secure) | ~$0.69/hr | **$504** |
| Serverless **Active**(常热) | $0.00013/s = $0.47/hr | **$343** |
| Serverless **Flex**(归零) | $0.00019/s = $0.68/hr | **按用量** |

用真实负载算 Flex:一份 28 页报告 ≈ 35 秒 GPU 时间。

| 每天报告量 | 每天 GPU 时间 | Flex 月成本 |
|---|---|---|
| 20 份 | ~12 分钟 | **~$4** |
| 100 份 | ~1 小时 | **~$20** |
| 500 份 | ~5 小时 | **~$100** |

冷启动本身几乎不要钱:400 秒 × $0.00019 = **$0.076/次**。一天冷启 20 次也才 $1.5。

### 两个反直觉的结论

1. **Serverless Active($343)比 Community Pod($285)还贵。** 如果结论是「必须 7×24 热」,
   那 Pod 反而便宜 —— 但 Pod 没有队列、没有自动重试、封顶一张卡,而且 Community 是别人的机器。
2. **成本压根不是决策点。** 我们这种突发负载,Flex 比常开便宜一到两个数量级。
   真正的决策点是:**用户能不能忍受空闲之后第一页等 6-7 分钟。**

所以 §6(攻 301 秒冷启动)不是为了省钱,是为了让 Flex 归零变得可用。
中间路线:RunPod 的 **idle timeout 可配**(默认 5 秒),拉到 10 分钟就能让一个工作时段内的
连续使用保持热,只有当天第一次付冷启动代价。

> 运维坑:**连续 7 天没有请求,RunPod 会把 max workers 自动设为 0**。淡季要注意。

### 对 C# 端的影响

Pod 是同步 HTTP,最简单。Serverless 要 create → poll:
- `/runsync` 的结果窗口只有 **1 分钟(最长 5 分钟)** —— **冷启动 6-7 分钟会超窗**,不能用
- 必须 `/run` + 轮询 `/status`(结果保留 30 分钟)

好消息:`GlmOcrClient.cs` 为 Replicate 写的就是 **create → poll → unwrap** 这个形状,
换成 RunPod 只是换 URL 和字段名,不用重写。

---

## 2c. 运行策略 —— **Flex + FlashBoot**(现行方案)

> **2026-08-11 改版。** 原方案是「定时脚本翻 `workersMin`」,但 Lotion 在 RunPod
> organization 里**不是 owner**,拿不到端点写权限。改走 Flex + FlashBoot ——
> 不需要任何端点写权限,而且如果 FlashBoot 有效会更便宜。
> 旧方案存档在 §2c-old,只有「FlashBoot 实测无效」**且**「拿到端点写权限」时才复活。

### 端点配置

| 设置 | 值 | 说明 |
|---|---|---|
| Endpoint Type | **Queue** | LB 端点无 worker 时 2 分钟超时,冷启动 6-7 分钟必挂 |
| GPU | **24 GB**(L4/A5000/3090) | Phase 1 实测档位 |
| **Active Workers** | **0** | 永远 0,不再翻开关 |
| **Max Workers** | **1** | 一张卡;突发**排队**不扩容 |
| **Idle Timeout** | **600s** 起步 | 测完 FlashBoot 再定 |
| **FlashBoot** | **Enabled** | 现在它是主力,不是附赠 |
| Env | `MAX_CONCURRENCY=4` | 占位,待 ladder 标定 |

### 全部押在 FlashBoot 上 —— 所以它必须先被验证

FlashBoot 是 **CRIU 式进程快照**:worker 缩到 0 时把整个进程树的状态(含 CUDA 显存)
快照下来,下次直接复活。官方原话:

> *FlashBoot only snapshots state that already exists in the worker process when it
> scales to zero.*

我们的 worker 缩零时 vLLM **已经编译完、CUDA graph 已经捕获完** —— 那 301 秒的成果就在
进程状态里。快照若抓到了,冷启动问题直接消失。

**但不能假设。** 两条反证:
1. 公开 issue「Very slow cold starts even with flashboot」,而且正好是 **vLLM worker**
2. 快照存**宿主机本地**,下次落到别的机器就没有

所以 Phase 2 的**第一件事**是 `MODE=coldstart` —— 打一次、等过 idle timeout、再打一次,
读 RunPod 返回的 `delayTime`。判定标准写死在脚本里:

| 最坏复活延迟 | 结论 |
|---|---|
| **< 60s** | FlashBoot 有效 → Idle Timeout 降到 60s,最省钱的配置 |
| 60-180s | 有帮助但不免费 → Idle Timeout 保持 600s |
| **> 180s** | **无效**(完整启动 ~400s)→ 要么拉长 Idle Timeout 盖住工作日,要么复活 §2c-old。**绝不能让用户等 6 分钟出第一页** |

### 成本

| 方案 | 月成本 | 前提 |
|---|---|---|
| **Flex + FlashBoot + idle 600s** | **~$27** | FlashBoot 有效 |
| Flex + idle 1 小时(硬扛) | ~$180 | FlashBoot 无效时的退路 |
| §2c-old 定时 Active | ~$115 | 需端点写权限(现在没有) |

FlashBoot 若有效,这条路**比原方案便宜 4 倍,且零脚本零权限**。

### 权限:降到最低档

| | §2c-old | 现行 |
|---|---|---|
| 需要的 API key 权限 | 改端点(`PATCH workersMin`) | **只要能跑 job** |

调用端点本身就要认证,这个绕不过 —— 生产的 C# 端也需要同一档权限。

---

## 2c-old. 定时 Active 方案(存档,当前不用)—— 一张卡定死 + 白天热晚上冷

### 端点配置

| 设置 | 值 | 作用 |
|---|---|---|
| `workersMax` | **1** | **一张卡,永远不扩容**。突发请求排队,不加机器 |
| `workersMin` | 白天 **1** / 夜间 **0** | 白天常驻(Active 费率),夜间归零 |
| `concurrency_modifier` | ladder 标定值 | 这一个 worker 同时接几个 job,其余 RunPod 排队 |
| Idle timeout | 60s | 只在夜间生效(白天 workersMin=1 不会缩) |

`workersMax=1` 就是「自己排队」—— RunPod 的 queue 端点本来就会把超出 worker 处理能力的
请求压在队列里,把 max 锁死成 1 之后它**只能排队,不能加机器**。

### 白天热 / 夜间冷 怎么实现

**RunPod 没有内建定时功能**,要自己调 API 翻 `workersMin`:

```
PATCH https://rest.runpod.io/v1/endpoints/{endpointId}
{ "workersMin": 1 }
```

实现:`C:\RunPodGLMOcr\warm_schedule.py`,**每 5 分钟跑一次**(Task Scheduler 一个触发器)。

设计要点:
- **自愈**:脚本从时钟算出「应该是什么状态」再对账,不存状态、不配对开关任务。漏跑一次,下一次自动补回来。比「08:00 开 / 19:00 关」两个任务稳 —— 那种一旦漏一次就整晚烧钱。
- **check-then-patch**:先 GET 再比,值没变就不写。Replicate 上同值 PATCH 会触发 release 滚动
  **替换掉正在跑的实例**(白白冷启动一次)。RunPod 是否一样**未验证**,所以干脆不发无谓的写。
- **提前预热**:冷启动 6-7 分钟,所以 `WARM_FROM = 07:50` 而不是 08:00 —— 约 07:57 热好。
- 顺手把 `workersMax` 也钉回 1,防止有人在控制台改了。

### 成本

24GB 档:Active **$0.00013/s**、Flex **$0.00019/s**(Active 是官方 40% 折扣档,建档时在控制台确认一下)。

| 方案 | 月成本 |
|---|---|
| **本策略(周一至五 07:50-19:00)** | **~$115** |
| 加上周六(`WARM_ON_WEEKENDS=True`) | ~$138 |
| Serverless Active 7×24 | $343 |
| Pod 常开(Community) | $285 |
| Pod 常开(Secure) | $504 |

夜间的零星请求走 Flex,一次冷启动 $0.076,忽略不计。

### 这个策略的代价(要认)

**一张卡 = 吞吐硬顶 0.94 pages/s。** 28 页报告 ≈ 30 秒。但如果 10 份报告同时提交:

| 同时提交 | 总页数 | 排队跑完 |
|---|---|---|
| 1 份 | 28 | ~30 秒 |
| 5 份 | 140 | ~2.5 分钟 |
| 10 份 | 280 | ~5 分钟 |

排队意味着**最后一份要等前面全部跑完**。如果哪天发现审计旺季这个等待不可接受,
把 `workersMax` 从 1 改成 2-3 就能弹起来(只在真正用到时才多花钱)——
**但那是改一个数字的事,现在不用为它设计。**

### 待验证(建好端点第一天要看)

1. PATCH `workersMin` 0→1 是**只启动 worker**,还是会**滚动替换**?后者会白白多一次冷启动
2. `workersMin=1` 是否**自动**按 Active 费率计费(文档一处说折扣「now live」,另一处提到
   sales inquiry)—— 在账单里核对
3. 连续 7 天无请求 RunPod 会把 max workers 自动设为 0 —— 我们的脚本每 5 分钟钉回 1,
   应该能挡住,但要确认

---

## 3. 排队:三层,自上而下

你问的「他不会 queue 吗」—— 源码证实:**glmocr 有队列,但在错的层级**。

`glmocr/pipeline/_state.py:34-35` 的 `page_queue` / `region_queue` 是**一次 parse 调用内部**的
流水线队列(这一个请求里的多页)。跨 HTTP 请求**没有任何准入控制** —— 搜遍全包,GPU 前面没有
Semaphore 也没有 Lock(只有 `_results_lock` 等数据结构锁)。

加上 Flask `threaded=True`:**N 个并发请求 = N 条独立流水线同时抢显存**,抢输的 OOM。
而 `_workers.py:263-281` 把 OOM 当永久失败吞掉:

```python
except Exception as e:
    logger.warning("Layout detection failed for pages %s, skipping batch: %s", ...)
    for page_idx in batch_page_indices:
        state.layout_results_dict[page_idx] = []   # 空结果
    return                                          # 不重试、不等待、仍算成功
```

所以排队要我们加。三层:

### 层 1 — RunPod 平台队列(零代码)

```python
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda current: MAX_CONCURRENCY,   # 固定值,不动态调
})
```

`MAX_CONCURRENCY = 6`。Ladder 实测:**N≤8 全清,N=12 崩(6/12 空,日志正好 6 条 OOM)**。
取 6 留一档余量。超出的请求 RunPod 替我们排在队列里。

> 不用文档示例里那种「按请求速率动态加减」的 modifier —— 我们的上限由**显存**决定,是个常数,
> 动态调只会自己撞悬崖。

### 层 2 — 容器内信号量(保险)

```python
_gate = asyncio.Semaphore(MAX_CONCURRENCY)
async def handler(event):
    async with _gate:
        ...
```

防住任何绕过平台队列的路径(重试、健康检查、将来的 LB 端点)。

### 层 3 — 空结果毒丸(必须留)

空结果和成功在 HTTP 层**无法区分**,前两层不能当唯一防线。见 §5。

---

## 4. 适配器 — 补齐 Z.ai 契约

自建返回的信封与云端有五处差异,全部实测确认:

| 项 | 自建 | Z.ai 云端 | 处理 |
|---|---|---|---|
| **`bbox_2d`** | **0-1000 归一化** | **绝对像素** | **`x_px = round(x*W/1000)`** ← 最关键 |
| `polygon` | 有 | 无 | 删 |
| 元素 `height`/`width` | 无 | 有(原图尺寸) | 补 |
| `data_info.pages` | `[]` | `[{height,width}]` | 用输入图尺寸填 |
| `usage` | `{}` | `{prompt_tokens, completion_tokens}` | 补 0(自建不按 token 计费) |

bbox 换算的依据是源码写死的:
- `glmocr/layout/base.py:42` → self-hosted = normalized (0-1000)
- `glmocr/api.py:353` → MaaS(云)= absolute pixel coordinates

验证:Directors' Report 页逐元素对黄金样本,**差 1-4px**(纯量化舍入)。
没有这一步,`ReportStitchOcr` 的缝合、`HeaderRecovery`、`BboxAspect` 会**静默错位**,而所有字段
看起来都完全正常。

---

## 5. 空结果的判定 —— 不再猜

**这是本设计里唯一的新机制。** 空 markdown 有两种成因,后果天差地别:

| 成因 | 该怎么办 |
|---|---|
| layout CUDA OOM 跳过批次 | **必须失败**,让 RunPod 重试。静默返回空 = 审计数据出错且无人知道 |
| 页面本来就是空白页 | 正常返回空信封 |

两者在 HTTP 层一模一样。但 **handler 和 glmocr 在同一个容器里** —— 可以直接读它的日志判定,
不用启发式:

```python
LOG = "/var/log/glmocr.log"

def _log_size():
    try:    return os.path.getsize(LOG)
    except: return 0

def _oom_since(offset):
    """Did the layout stage skip a batch since `offset`? Deterministic, not a guess."""
    try:
        with open(LOG, "rb") as f:
            f.seek(offset)
            return b"skipping batch" in f.read()
    except Exception:
        return False        # can't read -> don't claim OOM
```

判定流程:

```
mark = _log_size()
body = parse(image)
if md is empty:
    if _oom_since(mark):        -> raise  (硬失败,RunPod 重试;层 1/2 失效的信号)
    else:                       -> 重试一次
        if 第二次仍空:
            if _oom_since(mark) -> raise
            else                -> 返回空信封 + warning 字段(真空白页)
```

> 为什么对空白页不硬失败:扫描件里空白页很常见,硬失败会让整条流水线停。
> 为什么对 OOM 必须硬失败:**假成功的审计数据是不可见的错误**,比可见的失败危险得多。

---

## 6. 冷启动的 301 秒 —— 用实验解决,不用猜

```
Loading weights took 0.43 seconds          <- 模型加载
Loading weights took 0.27 seconds
init engine (profile, create kv cache, warmup model) took 301.12 seconds   <- 全在这
```

**把权重烘进镜像救不了冷启动**(权重只占 0.7 秒)。301 秒是显存 profiling + KV cache 分配 +
CUDA graph 捕获 + torch.compile。

Phase 2 要按序试这三个,**每个都量启动时间和吞吐**,不预设结论:

| 实验 | 预期代价 |
|---|---|
| **A. 烘焙 vLLM compile cache**(`VLLM_CACHE_ROOT` 指向镜像内目录,构建时跑一次) | 零运行代价,若命中就是纯赚 |
| **B. `--enforce-eager`** | 跳过 CUDA graph 捕获,启动大幅缩短,**推理变慢**(要量) |
| **C. 砍 `cudagraph_capture_sizes`** | 只捕获我们实际用的批次尺寸,A 和 B 之间的折中 |

验收:启动 < 2 分钟且吞吐不低于 0.85 pages/s → LB 端点重新可选。

---

## 7. 成本(24GB L4/A5000 档,实价)

| | 单价 | 说明 |
|---|---|---|
| Flex(可归零) | **$0.00019/s = $0.68/hr** | 只在跑的时候计费 |
| Active(常驻) | **$0.00013/s = $0.47/hr** | 归零不了,$343/月 |

**关键认识:冷启动的代价是延迟,不是钱。** 一次 400 秒冷启动 = **$0.076**,可以忽略。
但让用户等 7 分钟出第一页不可接受。

所以策略取决于 §6 的结果:
- 冷启动压到 <2 分钟 → **纯 Flex + 归零**,按用量付费
- 压不下去 → 要么 1 个 Active worker($343/月),要么把 idle timeout 拉长到覆盖工作日的间隙

28 页报告 ≈ 35 秒 GPU 时间 ≈ **$0.007**(不含冷启动)。

---

## 8. BDD 验收场景

**GIVEN** 一张真实 3166×4096 审计页
**WHEN** 提交给 serverless 端点
**THEN** 返回 `md_results` 非空、`layout_details` 元素齐四字段、`native_label` ∈ {paragraph_title, table, text}
**AND** `bbox_2d` 是**像素**(最大 x 接近页宽,不是 ≤1000)
**AND** 与 Z.ai 黄金样本逐元素比对,bbox 差值 ≤ 5px

**GIVEN** 端点并发上限设为 6
**WHEN** 同时提交 20 个请求
**THEN** 20 个全部成功(其余在 RunPod 队列里等)
**AND** 容器内 glmocr 日志 `grep -c 'out of memory'` = **0**

**GIVEN** 人为把 `--gpu-memory-utilization` 调到 0.85 制造 OOM
**WHEN** 提交一页
**THEN** handler **抛错**(job failed),**不**返回空信封
**AND** 错误信息里指明是 layout OOM

**GIVEN** 一张真正的空白页
**WHEN** 提交
**THEN** 返回空 `md_results` + `warning` 字段,**不**抛错

**GIVEN** 一个冷的(scale-to-zero)端点
**WHEN** 提交第一个请求
**THEN** 在 §6 验收线内返回,且**不**因超时失败

---

## 8b. 镜像怎么构建 —— 走 GHCR,不走 RunPod GitHub 集成

两条路都能用,我们选第二条:

| | RunPod GitHub 集成 | **CI → GHCR(现行)** |
|---|---|---|
| 谁构建 | RunPod | GitHub Actions |
| 需要什么权限 | **把 GitHub 连到 RunPod = owner 级操作** | 只要在建端点时填一个镜像 URL |
| 镜像存哪 | RunPod 自有 registry | GHCR **私有** package |
| 版本可控性 | 跟分支走 | **按 commit SHA 打标签**,端点钉死某个 SHA |

选 GHCR 的两个理由:
1. **Lotion 在 organization 里不是 owner**,连 GitHub 到 RunPod 这一步大概率做不了
2. 镜像里烘了 `selftest_page.png`,那是**真实客户审计页**,镜像**必须私有**。
   GHCR 私有 package 免费,而且用内置 `GITHUB_TOKEN` 认证,不用额外管密钥

实现:`.github/workflows/build.yml`。push 到 main 且动了 Dockerfile/handler/start.sh/
runner.py/selftest_page.png 时自动构建,推两个 tag:`:latest` 和 `:<commit-sha>`。

> **端点要钉 SHA tag,不要钉 `:latest`。**「现在到底跑的是哪个镜像」不能靠猜 ——
> 这正是 Replicate 那边 deployment 停在旧版本、白跑三次 warm6 的原因。

### ⚠️ 私有 GHCR 需要在 RunPod 配 registry 凭证

RunPod 拉不了私有 ghcr.io 镜像,除非先配好凭证:

**Settings → Registry Credentials → 新建一条**
- Registry: `ghcr.io`
- Username: 你的 GitHub 用户名(`lotion-abot`)
- Password: 一个 GitHub **PAT**,勾选 `read:packages` 权限

建端点时在 Container Image 填 `ghcr.io/lotion-abot/runpodglmocr:<sha>`(**小写**,GHCR 不收大写),
并选上这条凭证。

> 粘贴时别带空格 —— 这是这一步最常见的失败原因。

---

## 9. 文件清单与构建顺序

```
C:\RunPodGLMOcr\
  Dockerfile          # 单环境,钉死版本,烘两个模型,构建时固化 merged config
  handler.py          # runpod.serverless.start + 信号量 + 适配器 + OOM 判定
  start.sh            # 容器入口:起 vLLM -> 起 1 个 glmocr -> 起 handler
  runner.py           # 复用(构建时跑一次)
  runpod_test.py      # 复用(加一个 serverless 端点传输模式)
```

构建顺序:

1. `Dockerfile` + `start.sh`,本地 `docker build` 起来,**用 `runpod_test.py` 打通** —— 这一步不碰 RunPod
2. 加 `handler.py`,本地用 `runpod` SDK 的 test 模式验 `event["input"]` 契约
3. 推镜像 → 建 Queue 端点 → `concurrency_modifier=6` → 跑 §8 全部场景
4. §6 的三个冷启动实验
5. **全绿之后**才动 C#:`GlmOcrClient.cs` 加 RunPod provider(create → poll → unwrap)

> 第 5 步之前不碰 C#。这是既定规矩。

---

## 10. 钉死的版本(照抄 Phase 1 实测)

```
python            3.12
vllm              0.19.1
glmocr            0.1.5
transformers      5.15.0
torch             2.10.0
torchvision       0.25.0
flashinfer-python 0.6.6
```

vLLM 启动参数(实测被接受,dtype 自动 bfloat16):

```
--model zai-org/GLM-OCR --served-model-name default glm-ocr
--max-model-len 32768 --gpu-memory-utilization 0.70
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

> util 从 Phase 1 的 0.55 **提回 0.70** —— 因为只剩 1 个 glmocr 进程(省 3.8 GiB)。
> 提上去之后**必须重跑 ladder 重新标定 OOM 悬崖**,再据此定 `MAX_CONCURRENCY`。
> 现在写的 6 是 util=0.55 / 1 进程下测出来的,换了 util 就不作数。

---

## 11. 待定(需要 Lotion 拍板)

1. **机房**:现在这个 pod 在 EU-RO-1(罗马尼亚)。生产要不要挑靠近马来西亚的区?
   端到端延迟还没量,量完再定。
2. **Secure vs Community**:真实客户审计文件建议 Secure Cloud(RunPod 自有机房),
   Community 是第三方机器。价差要确认。
3. §5 的空白页策略:我选了「OOM 硬失败 / 空白页软返回」。如果你要更保守(任何空都硬失败),
   改一行就行。
