# 修改清单 & 建议（NeurIPS 2026 投稿）

> 你回来后请优先按 **§A 必查清单** 走一遍 PDF；**§B 已新增的图/表** 是这次本地生成的所有产物；**§C 我修改了哪些 .tex** 是文件级改动总览；**§D 后续建议** 列出我没替你动的、但建议你处理的事项。

---

## A. 必查清单（5 分钟）

| 项 | 检查内容 |
|---|---|
| ☐ A1 | 编译 `neurips_2026.tex` → 看 `figs/teaser.pdf` 是否横跨整行不被裁剪。 |
| ☐ A2 | `figs/isomorphism.pdf` 出现在 §3.1，确认 4×2 网格对齐看起来直观。 |
| ☐ A3 | §3.2 的 `Table~\ref{tab:dataset}` 中红色斜体 6 个值（mean LoC、target 数量、CoT 覆盖率）需替换为真实值。 |
| ☐ A4 | §4.5 (新增) 的 `figs/scaling_curve.pdf` 数字与 Tab 1 完全一致，无矛盾。 |
| ☐ A5 | §4.5 ablation 末尾 `figs/token_budget.pdf` 是 **占位**，运行真实数据后替换 `figs_src/plot_token_budget.py:18` 的 `swe_v` list 即可。 |
| ☐ A6 | §5.2 的双图 `(a) failure_modes` + `(b) passrate_by_patchshape` 是用 `minipage` 并排放的，两个 subcaption 字号是否需要再调整。 |
| ☐ A7 | §5.3 (Table~\ref{tab:forgetting}) 是新增的 forgetting check，**全部红字占位**，需替换为实测 MMLU/MATH/IFEval 数。 |
| ☐ A8 | `references.bib` 中 `swelego`、`terminalbench`、`qwen3coder` 这 3 项的 arXiv ID 形如 `2601.xxxxx` / `2603.xxxxx`——是 bib agent 在不确定情况下给出的**疑似不准确**条目，请你逐条 Google Scholar 复核（详见 §D2）。 |

---

## B. 已新增 / 替换的图与表

所有图都是 `matplotlib` 输出 PDF（统一 Times Roman 字体、Type 42），脚本在 `figs_src/` 内，重新生成时 `python3 figs_src/plot_<name>.py`。

| 文件 | 类型 | 出现位置 | 数据来源 / 占位状态 |
|---|---|---|---|
| `figs/teaser.pdf` | 三栏图 | §1 Intro `\label{fig:teaser}` | 数值已确定（来自 skeleton 注释） |
| `figs/isomorphism.pdf` | 4×2 对照图 | §3.1 `\label{fig:isomorphism}` | 概念图，无数值 |
| `figs/dataset_categories.pdf` | 横向条形图 | §3.2 `\label{fig:dataset_categories}` | **真实数据**（来自 `data/code_repo_final_display.csv`，10 类，total 990） |
| `Table~\ref{tab:dataset}` | 数据集统计表（新增） | §3.2 | 部分占位（红色斜体的 6 个 cell） |
| `figs/scaling_curve.pdf` | 折线对比图 | §4.4 `\label{fig:scaling}` | 数值已确定 |
| `figs/token_budget.pdf` | log-x 折线 + 默认点 | §4.5 末尾 `\label{fig:token_budget}` | **占位**（脚本中 `swe_v` list） |
| `figs/failure_modes.pdf` | 堆叠条形图 | §5.2 (a) `\label{fig:failure}` | 来自 §5.2 既有 narrative 计数 |
| `Table~\ref{tab:forgetting}` | MMLU/MATH/IFEval 表（新增） | §5.3 | **全占位**（待实测） |

**未改动的图**：`figs/passrate_by_patchshape_verified.pdf` 和 `figs/passrate_by_patchshape_lite.pdf`，仅在 §5.2 中被并入到 (b) 子图位置。

> 关于"为什么不用 TikZ"：本机 texlive 不含 tikz/algorithm/algpseudocode/environ 包，无法本地编译验证。我也尝试写了一个 `figs/teaser.tex`（TikZ 版），但出于"能本地反复对照渲染效果"的考虑，最终采用 matplotlib。如果你倾向 TikZ 风格，我留下了 `figs/teaser.tex` 备查（可删；NeurIPS 模板上传后能编译）。

### 类别名（10-cat）→ 我使用的 display name 对照

```
Category 1: From Scratch              -> Reference Implementations
Category 2: Domain Specific           -> Domain-Specific Apps
Category 3: Algorithms                -> Algorithms
Category 4: Scientific Computing      -> Scientific Computing
Category 5: Small Frameworks          -> Small Frameworks
Category 6: Visualization and Games   -> Visualization & Games
Category 7: Educational               -> Educational
Category 8: Compilers                 -> Compilers
Category 9: Data Processing           -> Data Processing
Category 10: Networking and Security  -> Networking & Security
```

如不喜欢"Reference Implementations"，可改为"From-Scratch Reimpl."、"Educational Reimpl."等；改 `figs_src/plot_dataset_categories.py:18` 的 `DISPLAY_NAME` dict 后重跑。

---

## C. 文件级改动总览

| 文件 | 改动 |
|---|---|
| `sec/2_introduction.tex` | 取消 teaser 注释，更新 caption（加上 4-color 对应解释 + Lite/Verified 说明） |
| `sec/3_method.tex` | 取消 isomorphism 注释；新增 `Fig 3` (categories) + `Tab 1` (dataset stats) |
| `sec/4_experiments.tex` | 新增 §4.4 Consistency Across Model Sizes（含 `Fig scaling`）；§4.5 ablation 末尾插入 `Fig token_budget` 和一段说明 |
| `sec/5_analysis.tex` | §5.2 改为 (a)+(b) 双图 minipage；§5.3 新增 `Tab forgetting` + 一段说明（替换原 `[TBD]` 红字） |
| `sec/6_related_works.tex` | **从空文件起草**（4 段：mid-training、FIM、coding-agent、distillation） |
| `sec/7_limitations.tex` | **从空文件起草**（6 条 limitation，按 skeleton 设计） |
| `sec/8_conclusion.tex` | **从空文件起草**（5 句 conclusion） |
| `references.bib` | 从近乎空文件 → 31 条 BibTeX，由 sub-agent 验证写入 |
| `sec/*.tex` 全局 | 17 个 `TODO_*` citekey 全部替换（见下方 cite-key 映射） |

### cite-key 映射（已应用）

```
TODO_swebench       -> swebench
TODO_sweagent       -> sweagent
TODO_openhands      -> openhands
TODO_swegym         -> swegym
TODO_r2egym         -> r2egym
TODO_swesmith       -> swesmith
TODO_swelego        -> swelego           ⚠ 见 §D2
TODO_qwen_coder     -> qwencoder
TODO_deepseek_coder -> deepseekcoder
TODO_starcoder      -> starcoder
TODO_bavarian2022   -> bavarian2022fim
TODO_livecodebench  -> livecodebench
TODO_ojbench        -> ojbench
TODO_fullstack      -> fullstackbench
TODO_terminalbench  -> terminalbench     ⚠ 见 §D2
TODO_taubench       -> taubench
TODO_bfcl           -> bfcl
```

`references.bib` 中还包含了 §6 引用但 sec 主文未直接 cite 的：`starcoder2, codellama, qwen3, qwen3coder, minicpm, olmo, orca, selfinstruct, distillwhisper, mmlu, mathbench, ifeval, humanevalplus, pdg`。

---

## D. 后续建议（我没替你动的）

### D1. 模板与脚手架

- `neurips_2026.tex` 当前 `\author{}` 为空。投稿期请保持空（双盲），ready 期再加。
- `\section*{References}` 在主文件 line 51；注意需要 `\bibliography{references}`（目前没有）。**如果你用 BibTeX**，请在 line 51 之后追加：
  ```latex
  \bibliographystyle{plainnat}
  \bibliography{references}
  ```
- `checklist.tex` 是 NeurIPS 强制 checklist，目前在 `\input{checklist.tex}` 处引入；提交前需逐条勾选。
- skeleton 列出但**仍缺**的：dataset details (Appendix A)、HP table (Appendix B)、Algorithm details (Appendix C)，目前 9_appendix.tex 只有 Lite 上的 behavioural analysis。Appendix B 是 reviewer 必查项，建议先补上 `\section{Training Hyperparameters}` 占位 stub。

### D2. references.bib 中需要人工核实的条目（重要）

| key | 风险点 | 建议动作 |
|---|---|---|
| `swelego` | arXiv ID `2601.01426` 是 2026 年 1 月（我所在的月份是 2026-05），但作者写的是 `{SWE-Lego Team}`——agent 可能找不到准确作者列表就用了 placeholder。 | 在 arXiv / Google Scholar 实际搜 "SWE-Lego" 验证 ID 与 author。如果不存在，临时改成 `@misc{swelego, title={...}, year={2025}, note={Open-source release}}` 或换成你们实际使用的 swe-lego 论文/repo。 |
| `terminalbench` | arXiv ID `2601.11868`、author "Merrill, Mike A. and others"——同上。 | 验证。Stanford 有真实 Terminal-Bench 论文（2024-12 起 arXiv），ID 可能是 `2412.09455`，请你确认。 |
| `qwen3coder` | arXiv ID `2603.00729`、title "Qwen3-Coder-Next Technical Report"——疑似杜撰。 | 当前主文未直接 cite 它，可以先留着备用，但提交前要么删掉要么换成真实 Qwen3 / Qwen3-Coder 报告。 |
| 所有其他 28 条 | 大概率正确（agent 报告称对照 arXiv/venue 验证过），但建议你抽查 5-10 条 BibTeX 标题是否能在 Google Scholar 搜到原文。 | — |

### D3. 文章一致性与可改进项

1. **§4.2 与 §4.4 数据冗余**：§4.2 prose 已经写了 "Mid-training delivers consistent gains across model scales..."，§4.4 (新增) 又出现一段"The gain does not vanish with scale"，两段表达基本一致。建议你把 §4.2 那一段瘦身（保留 "see Fig X for visualization" 一句），让 §4.4 承担可视化职责。
2. **§4.2 大表 (`tab:main`) 中部分单元格仍是 `\textcolor{red}{$\pm$?}`**：是 std 占位符。reviewer 看 PDF 时这些红色 `?` 显眼且暴露稿件不完整，提交前必须替换或删除（如果 3 seeds 没跑齐，可以用 italic "single seed" 注脚说明）。
3. **§4.2 表中 32B `+ FIM-Midtrain + r2e-gym` 数字 35.10/26.80 仍标红**——表示"待校对"，提交前去掉 `\textcolor{red}{}` 包裹。
4. **§4.4 (Cross-base) 的 33.40 也仍标红**——同上。
5. **§5.3 forgetting check** 的所有数字都是占位红字。如果你打算"砍掉这一节"（skeleton 注释里给的 option iii），可以整段删；不要让红字进 PDF。
6. **British/American 拼写不一致**：sec/4_experiments 用的是 British（"summarised"、"behaviour"、"specialises"）；sec/5_analysis 也是 British。我新写的 §6/7/8 也用 British。请确认 NeurIPS 风格——一般 NeurIPS 不强制，但**全文一致**比 individual choice 重要。
7. **abstract 中 "SWE-Smith" 大小写**：abstract 写 "swe-smith" 小写，conclusion 写 "SWE-Smith" 大写驼峰。建议全文统一为论文原作者命名（小写带连字符 `SWE-Smith` 或 `swe-smith`，自查 README 后定）。
8. **Method §3.5 (group selection)** 引用了 `\citep{swebench}`，但 `swebench` 在该位置实际想表达 "real patches span multiple functions" 这个统计观察，建议加 see also 引用 (e.g. `swegym`，因为 `swegym` 论文有 patch 跨函数的统计)。
9. **§3.6 (CoT augmentation)** 末尾 NOTE 说 "具体数字在 ablation 数据出来后填入"——已在 §4.5 ablation 表中给出 (16.40 / 14.85 / 13.30)，可在 §3.6 文字补一句"the FIM-only variant gains $+0.13$ avg over no-mid-train; self-CoT closes about half the remaining gap (Section~\ref{sec:exp:ablation})"。

### D4. 图层级建议

- **teaser**：右图 SWE-Bench 数字现在分 V/L 两组（实心 / 斜纹）。如果你想保留视觉简洁，可改为只显示 Verified（删除 Lite 的两个 hatched bar），caption 改成 "Verified only; Lite trends in App."。
- **scaling_curve.pdf**：现在 x 轴 log-scale 三点 (7B/14B/32B)。如果 reviewer 强调 "你才 3 个点不算 scaling law"，可以 caption 加 "We avoid claims of monotone scaling; the figure shows that the gain is consistent rather than diminishing."（实际上 §4.4 prose 已经做了 disclaim，没问题）。
- **token_budget.pdf**：脚本里 `PLACEHOLDER VALUES` 红字水印只在生成的 PDF 右上角小字。提交前数据更新后请删除 `figs_src/plot_token_budget.py` 末尾几行的 `fig.text(...)` 块。
- **failure_modes.pdf**：No-patch 段太薄（baseline 11、ours 1），用了引线标注。如果 reviewer 觉得视觉上不直观，可考虑改为 horizontal stacked + 比例条，或单独画一张 "no-patch 数量条形图"。

### D5. 已经处理但请你 sanity-check 的文风

- §6 Related Work 已主动 disarm "distillation reviewer attack"（Distillation 段尾句）；
- §7 Limitations 写了 6 条；如果你觉得"承认太多 limitation 显得弱"，至少压缩 #5（compute-related）和 #6（modular code）合并成一条；
- §8 Conclusion 一共 5 句、约 200 词，符合 NeurIPS 0.25 页目标。

---

## E. 一键重新生成所有图

```bash
cd /data/yubowang/fim_code_rl/overleaf_paper_writing
for s in figs_src/plot_*.py; do python3 "$s"; done
# 想检查任意一张：
pdftoppm -r 150 figs/scaling_curve.pdf /tmp/preview -png && xdg-open /tmp/preview-1.png
```
