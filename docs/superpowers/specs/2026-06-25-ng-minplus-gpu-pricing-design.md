# NgMinPlusSolver — GPU ng-route min-plus 定价后端 设计文档

**日期**：2026-06-25  
**状态**：待实现  
**仓库**：`bucket-graph-spprc`

---

## 1. 背景与动机

### 1.1 问题设置

本设计在 `bucket-graph-spprc` 中新增**第二个定价后端**：与现有 `include/bgspprc` 的不规则 label-setting 引擎**并列**，求解同一个 SPPRC/ESPPRC（初等容量约束最短路定价子问题），但用规则化的 min-plus 状态张量在 GPU 上实现。

两个后端共享同一 `ProblemView` / CLI / benchmark 层，对外行为一致：

```
for ng in 8 16 24; do
    ./benchmarks/run_benchmarks.sh --mode $mode --ng $ng --timeout 120 \
        benchmarks/instances/spprclib
```

弧成本是 `ProblemView` 提供的标准 reduced cost：

```
ĉ_{uv} = cost[u,v] + nodeprize[v]
```

其中 `cost[u,v]` 是图的弧成本，`nodeprize[v]` 是定价时由上层（列生成 RMP）注入的对偶贡献，经由现有 `update_arc_costs()` 管线传入。min-plus 后端不感知对偶的来源，只消费已经合成好的 `ĉ`，与 labeling 后端完全对等。

### 1.2 现有方案的局限

- **Held–Karp subset DP**（`2^m` 状态）：精确但状态随 m 指数爆炸，m > 20 不可用。
- **C++ labeling / BucketGraph**：dominance 剪枝强，但 label 数量不规则，在 GPU 上 warp 发散、负载不齐。
- **Python min-plus B1/B2**：B1 精确但仍在 CPU 上串行；B2 有状态压缩但 GPU 版未完成、未接入统一的 ProblemView/CLI 层。

### 1.3 核心洞察：同一 Bellman 递推的两种实现

```
ESPPRC 的 Bellman 最优子结构
        │
        ├── 不规则 label-set 实现  → BucketGraph（现有）
        │     并行不规则，dominance 剪枝是强项
        │
        └── 规则 state-tensor 实现 → NgMinPlusSolver（本设计）
              状态预先定形，(min,+) 矩阵积天然 SIMT 并行
              GPU 上的规则负载是它的强项
```

两者是互补关系，不是替换关系：同一份 `ProblemView`、同一组 ng-set、同一条 `Path` 输出，B&P / benchmark 层可无感知切换后端。

---

## 2. 设计目标

1. **精确**：返回认证初等（elementary）且 reduced cost 最小的路径，或证明无负 rc 路径存在。与 label-setting 后端在精确性上等价。
2. **可扩展**：状态规模 `O(loads · 2^Δ · V)`，Δ 可控，打破 Held–Karp 的 `2^V` 天花板。
3. **算力饱和**：单实例内，每个载重层是一次 `2^Δ × V` 规模的 `(min,+)` matvec；ng=16/24 时该状态空间足以填满 SM，无需跨实例批量即可喂饱 GPU。
4. **零侵入**：CUDA 路径通过 `BGSPPRC_BUILD_CUDA` 可选；缺席时 header-only 核心不受影响。

---

## 3. 状态模型与递推

### 3.1 设备张量

```
f[ℓ][Π][v]  (device memory, fp64)
```

| 维度 | 含义 | 大小 |
|---|---|---|
| `ℓ` | 累计载重（load layer） | `Q+1`（容量上限+1） |
| `Π` | ng-记忆 bitmask | `2^Δ`（Δ = ng 宽度） |
| `v` | 当前顶点 | `n_vertices` |

初始：`f[0][0][depot] = 0`，其余 `+∞`。

### 3.2 递推

$$
f[\ell][\Pi][v] = \min_{(u \to v) \in A,\; u \notin \text{ng-forbid}(\Pi, v)} \Big\{ f[\ell - d_v][\Pi'][u] + \hat{c}_{uv} \Big\}
$$

其中 `Π' = ng_extend(Π, u→v)`（按 Meta-Solver §4.2.2 的 destination-marking 更新）。

因为 `d_v > 0`（每个客户需求严格正），载重严格单调递增 → 状态图是按 ℓ 分层的 DAG → **从小到大单遍扫完，无环**。

### 3.3 精度选择：fp64

`ProblemView` 的弧成本和对偶值均为 `double`。ng 松弛已引入 rc 近似；再叠加 fp32 累积误差会使 DSSR 的"判负"（`rc < θ ≈ −1e-6`）不可靠。采用 fp64，Ampere+ 架构的 fp64 吞吐（1/2 of fp32 peak）在定价核上可接受。

---

## 4. "分层块状"执行策略

### 4.1 分层（hierarchical）

逐载重层 ℓ = 0, 1, …, Q 推进，每层是一次独立的 `(min,+)` matvec：

```
for ℓ = 1 to Q:
    launch load_layer_kernel(f[ℓ-d:ℓ], cost_block, ng_mask)
```

层间依赖严格单向，天然适合 CUDA stream 流水：层与层之间通过 stream 重叠 kernel 启动与访存，是单实例内并行的主力来源。

### 4.2 块状（block-parallel min-plus）

每层内，将 `(Π × v)` 状态空间切成 tile（ng-block × vertex-tile）。对整块越界（`ĉ = +∞`）的 tile 整体跳过，节省算力但不破坏精确性（跳的全是 `+∞` 贡献）。

Block skip 条件（B1 block-mask 思路）：若某 (ℓ, Π) 组合不可达，则 skip 整个 tile。

### 4.3 Kernel grid 结构

```
grid  = (n_ng_blocks, n_load_tiles)
block = (WARP_SIZE × n_warps_per_block)
```

每个 thread 负责一个 `(Π_chunk, v)` cell，执行一次 `(min,+)` 累积。ng=16/24 时 `n_ng_blocks` 已足够大，单实例即可占满 SM。

---

## 5. ng-route 松弛 → DSSR 增广 → 认证精确

### 5.1 DSSR 控制流（host 端）

```python
build ĉ blocks for the instance

loop:
    launch grid                  # device: 全量 (min,+) 张量推进
    route ← backtrack optimal    # host
    cyc = find_first_cycle(route)
    if cyc is None:
        return route             # 路径初等，rc 认证精确（或证明无负 rc 列）
    else:
        augment_ng(cyc)          # 引发环路的节点 ng-set 扩张，rebuild device mask
```

DSSR 是**单实例内**的迭代：每轮在当前 ng 松弛下解一次张量，若最优路有环则扩张闯环节点的 ng-set 重解，直到无环。这与批量无关——它是 ng→exact 的内在过程。

### 5.2 增广粒度：per-node ng-set

DSSR 只扩张**引发环路的节点**的 ng-set，其他节点不受影响。全局统一扩大 Δ 会让状态无差别膨胀，浪费张量稀疏性。

Per-node 状态存在 host 端的 `ng_augmentation_table[v]`，每轮增广后重建 device 侧 mask。

### 5.3 正确性论证

1. kernel 始终返回 ng-松弛下的最优路 → reduced cost 是 ESPPRC 真最优的**下界**。
2. 若返回路**无环**，它在初等可行域内，且 rc ≤ 真最优 → 即为真最优（下界 = 真值）。
3. 每次增广**严格收紧**松弛空间（ng-set 单调增大，最终 = full set = Held–Karp），状态有限 → 循环**有限步终止**。
4. 终止时：要么拿到认证初等的负 rc 列，要么证明无负 rc 列 → **CG 精确性保持**。

### 5.4 Backtrack 位置：host

最优路回溯是 O(V) 串行链表追踪，数据量极小。放 host 端，避免引入 device→host sync barrier 的复杂性；device 专注张量推进。

---

## 6. 单实例内的并行来源

### 6.1 算力从哪来

无需跨实例批量。单个 SPPRC 实例内部的并行有三层，逐层填满 SM：

1. **层内 `(Π × v)` matvec**：每个载重层是 `2^Δ × V` 规模的 `(min,+)` 矩阵积，ng=16 即 `65536 × V`、ng=24 更大——这是主力并行宽度。
2. **load-layer stream 流水**：相邻载重层通过 CUDA stream 重叠 kernel 启动与访存。
3. **block skip 稀疏性**：不可达 tile 整体跳过（§4.2），把算力集中在真正活跃的状态上。

### 6.2 与 ng 宽度的关系

`run_benchmarks.sh --ng 8/16/24` 三档恰好覆盖从"勉强占满"到"充分饱和"的区间：ng=8（`2^8=256` ng-mask）SM 占用偏低，ng=16/24 是 min-plus 后端相对 labeling 的优势区。对拍在所有三档上进行，吞吐评估聚焦 ng=16/24。

---

## 7. 与现有代码库的集成

### 7.1 共享部分（唯一真相源）

| 共享对象 | 来源 | 说明 |
|---|---|---|
| `ProblemView` | `include/bgspprc/problem_view.h` | 图拓扑、弧端点、容量上限 |
| ng-neighborhood sets | `NgPathResource` / `instance_io` | Meta-Solver §4.2.2 的 ng-set 结构 |
| 弧成本 / 对偶更新 | `update_arc_costs()` 管线 | ĉ 由上层传入，两个 solver 共用同一接口 |
| `Path` output type | `BucketGraph<Pack>::Path` | 输出路径格式一致，benchmark / B&P 层无感知切换 |

### 7.2 NgMinPlusSolver 自有部分

- 设备张量（`f`、`cost_block`、`ng_mask`）
- CUDA kernel（`ng_minplus_layer_kernel`）
- Host 端 DSSR 控制循环 + ng augmentation table

### 7.3 新文件布局

```
include/bgspprc/
    ng_minplus_solver.h          # 顶层 C++ host 类（CUDA 可选）
    ng_minplus_kernel.cuh        # CUDA kernel 声明
src/
    ng_minplus_kernel.cu         # CUDA kernel 实现
CMakeLists.txt                   # BGSPPRC_BUILD_CUDA 选项
```

`BGSPPRC_BUILD_CUDA=OFF`（默认）时，`.cu` / `.cuh` 不参与编译，header-only 核心不受影响。

### 7.4 CLI 集成入口

min-plus 后端作为 `bgspprc-solve` 的一个可选 solver，挂在与 labeling 同一条命令行上：

```
bgspprc-solve --solver minplus [--ng K] [...] <path>
```

- 单实例 `.graph` / `.sppcc` → 1 条 `Path`，与 `--solver labeling` 完全同形。
- 默认 `--solver labeling` 保持现状；`--solver minplus` 在编译开启 CUDA 时可用，否则报错退出。
- `run_benchmarks.sh` 通过既有的 `--mode` / `SOLVE` 机制选择后端，CSV 列结构不变。

逐问题对拍判据：`|rc_minplus − rc_labeling| < 1e-9`。

### 7.5 min-plus 专属统计字段

labeling 的 `n_buckets`、`n_dominance_checks` 对 min-plus 无意义。新增：

| 字段 | 含义 |
|---|---|
| `n_dssr_rounds` | DSSR 总轮数 |
| `n_augmentations` | ng-set 增广次数（总计） |
| `kernel_ms` | GPU kernel 总耗时（ms） |
| `gpu_occupancy` | 平均 SM 占用率 |
| `n_blocks_skipped` | block-mask 跳过的 tile 数 |

---

## 8. 构建配置

```cmake
option(BGSPPRC_BUILD_CUDA "Build CUDA ng-minplus backend" OFF)

if(BGSPPRC_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(bgspprc_cuda src/ng_minplus_kernel.cu)
    target_link_libraries(bgspprc_cuda CUDA::cudart)
    target_compile_features(bgspprc_cuda PUBLIC cuda_std_17)
    target_compile_definitions(bgspprc INTERFACE BGSPPRC_CUDA_AVAILABLE)
endif()
```

---

## 9. 验证阶梯

| 级别 | 内容 | 判据 |
|---|---|---|
| **a** | 单问题 vs Held–Karp `2^m`（Python baseline） | `\|rc_diff\| < 1e-9` |
| **b** | 单问题 vs C++ BucketGraph labeling（CLI `--solver` 对拍） | `\|rc_diff\| < 1e-9` |
| **c** | `spprclib` 全集 × ng∈{8,16,24} 逐实例对拍 | 逐个 `\|rc_diff\| < 1e-9` |
| **d** | ng=16/24、pricing 占主导实例的吞吐测试 | 每实例 pricing ms，加速比 vs labeling |

测试文件建议：
- `tests/test_ng_minplus_parity.cpp`（对拍 a/b/c）
- `tests/test_ng_minplus_throughput.cpp`（吞吐 d）

---

## 10. 开放决策记录（已定）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 精确性机制 | DSSR ng-augmentation loop | 状态有界 + fixpoint 认证精确 |
| 计算后端 | GPU CUDA | min-plus 张量是规则负载，labeling 不擅长的 SIMT 并行 |
| 并行来源 | 单实例内 `2^Δ × V` matvec + load-layer 流水 | ng=16/24 即可占满 SM，无需跨实例批量 |
| 状态模型 | 自有 gpu tensor，共享 cost/ng config | 不强行套 Resource/Pack 的不规则接口 |
| 精度 | fp64 | 对偶值 double，判负阈值 1e-6，fp32 累积误差不可接受 |
| Backtrack 位置 | host | 串行 O(V)，避免 device sync 复杂性 |
| 增广粒度 | per-node ng-set | 只扩张闯环节点，状态不无差别膨胀 |
| 集成方式 | `bgspprc-solve --solver minplus` | 与 labeling 同一 CLI/benchmark 层，`Path` 同形 |

---

## 11. 不在本 spec 范围内

- GPU 版双向（bidirectional）min-plus（可作后续扩展）
- 跨实例批量定价（多实例同时上卡；本 spec 聚焦单实例后端）
- 自适应 Δ 调度（当前 Δ 固定，DSSR 驱动 per-node 扩张）
- fp16/bf16 混合精度加速
- Multi-GPU sharding

---

## 12. 设计评审（概念澄清）

本节不引入新设计决策，只对 §1.3、§4、§5、§6 已落地的设计做一次概念层面的对齐，澄清三个容易被口号掩盖的要点，并说明三者如何协同。

### 12.1 两种实现是同一条 Bellman 递推

Labeling 与 min-plus DP 不是两个算法，而是同一条 ESPPRC Bellman 最优子结构（§1.3）的两种**状态表示与调度方式**：

- **不规则 label-set 实现**（BucketGraph）：状态（label）按需动态生成，数量不可预先预知，靠 dominance 剪枝压住规模——状态空间的形状是运行期才浮现的。
- **规则 state-tensor 实现**（NgMinPlusSolver）：状态空间 `f[ℓ][Π][v]` 预先定形，三个维度 `(Q+1) × 2^Δ × V` 在 kernel 启动前就静态已知——状态全集先铺定，递推在固定网格上做。

两者求解的是同一个最优子结构、同一组转移 `f[…] = min{f[…] + ĉ}`，差别只在状态被"如何表示"与"何时调度"，最优值与回溯路径必然一致。这正是 §9 逐实例对拍判据（`|rc_minplus − rc_labeling| < 1e-9`）在原理上必然成立的根本原因：对拍不是在赌两个实现碰巧相等，而是在验证两条实现没有偏离它们共同的 Bellman 定义。

### 12.2 并行的本质是 state-parallel（对状态并行）

min-plus 后端的并行**不是**对实例/批量并行（§6、§11 已明确不做跨实例批量），而是**对状态并行**：

- 因为状态张量预先定形，单个载重层的递推就是一次 `2^Δ × V` 规模的 `(min,+)` 矩阵积；"对每个目标状态取 min over 入弧"这件事天然映射成 SIMT——每个线程负责一个 `(Π, v)` cell，cell 之间在层内互不依赖（§4.3）。
- 并行宽度 = 该层活跃状态数 = `O(2^Δ · V)`，由状态空间几何**静态决定**、可提前预测，warp 内各线程做同构的 min 累积，无结构性 warp 发散。

这是规则张量相对不规则 label-set 的根本优势所在：labeling 的并行宽度取决于运行期 label 数量（不可预测、易负载不齐、warp 发散），而 min-plus 的并行宽度由 ng 宽度 Δ 和顶点数 V 这两个静态量直接给出。换言之，并行度是设计期可算的，而非运行期才知道的。

### 12.3 block mask 的角色是 pruning，不是 approximation

分层块状 Min-Plus 的 block skip（§4.2）容易被误读为"近似/丢状态"，实际上它在语义上**等价于 labeling 的 dominance/可达性剪枝**，是精确的：

- 被跳过的 tile 对应整块 `+∞`（不可达 `(ℓ, Π)` 组合）贡献，而 `+∞` 在 min 中是吸收元——跳过它们对 min 结果**零影响**。
- 区别只在剪枝的**粒度与时机**：labeling 是"逐 label 动态剪"（dominance 在生成时即时判定），min-plus 是"按 `(ℓ, Π)` tile 块状剪"（不可达整 tile 一次性跳过）。

因此 block mask 是把 labeling 那套不规则、逐点的剪枝，**重新表达成规则张量上的稀疏掩码**：既省算力（不算 `+∞` tile），又不破坏精确性（min 结果不变）。它是同一种"剪掉不可达/被支配状态"的思想在规则张量上的对应物，而非一种引入误差的近似手段。

### 12.4 三机制如何协同

ng-route 松弛、分层块状 Min-Plus 并行、DSSR 三者各管一件事，构成闭环：

- **ng-route 松弛**（§3.1）：Π 维只有 `2^Δ` 而非 `2^V`，把状态规模从 Held–Karp 的 `2^V` 指数压到可上 GPU 的常数幂——**控规模**。
- **分层（load layer 流水）+ 块状（tile 级 SIMT + block skip）**（§4）：把这个被压住的、规则的状态空间在 GPU 上铺满，让算力被饱和利用——**控吞吐**。
- **DSSR**（§5）：把 ng 松弛逐轮收紧到精确，终止时给出认证初等的负 rc 列或证明无负 rc 列——**认证精确**。

一句话收口：

> 用 ng-route 松弛确保状态规模可控，用分层块状 Min-Plus 并行确保算力被饱和利用。

前者控规模、后者控吞吐，DSSR 在二者之上把松弛收紧到精确——三者构成"可控规模 × 饱和算力 × 认证精确"的闭环。

---

*下一步：调用 writing-plans 生成实现计划。*
