# 待你处理的问题清单（中文）

下面列出本轮修订中我无法独自决定、需要你过目或补数据的事项。每条都标了优先级和位置。
英文细节版见 `REVIEW_FINDINGS.md`。

---

## 🔴 高优先级 — 需要补数据或必须改动

### Q1. License 表的逐项数量是占位符
- 文件：`sec/9_appendix.tex` 中的 `tab:licenses`。
- 现状：你说"具体的每个 license 的数量先占位，我后面修改"，所以表格里每个 license 的 count 我都改成了 `\textcolor{red}{TBD}`，"Total" 改成了 968。
- 需要你做：填入正式的逐项 license 数量。**注意**：附录 L76-83 紧接着的散文里还写着 "$\sim$79\% permissive, $\sim$5\% copyleft, $\sim$2\% Creative Commons"，这些百分比其实就是基于以前 990 仓库的统计；新的 968 仓库需要你重新算一下，或者临时把这几个百分比也用 `\textcolor{red}{TBD}` 占位。

### Q2. 主表里部分 std-band 是占位
- 文件：`sec/4_experiments.tex` 中 `tab:main`。
- 现状：4.1 节的 setup 一段写着 `\textcolor{red}{$\pm$?}` 是占位，待最终实验跑完替换。大多数行已经有真实 std，但 R2E-Gym (officially reported) 的 7B/14B 行的 $\pm$ 看起来仍像是估算。
- 需要你做：确认或替换为真实 std。

### Q3. 训练超参附录是 stub
- 文件：`sec/9_appendix.tex` 中 `Training Hyperparameters` 小节。
- 现状：明确写着 "Stub: hyperparameters and token budgets to be finalized at camera-ready time"。
- 需要你做：NeurIPS 审稿人会被这条直接拦下来，建议在投稿前补全（学习率、batch、token budget、warmup、optimizer、ZeRO/TP 配置等）。

---

## 🟠 中优先级 — 涉及表述/范围调整

### Q4. 跨 base-model 的说法只剩一个 8B 点
- 删除 32B 之后，跨"模型家族"的证据其实只剩 Qwen3-8B + SWE-Lego。
- 摘要里"the improvement holds across post-training pipelines and base-model families"现在略显夸张。
- 建议：**摘要**软化为 "across two post-training pipelines (R2E-Gym, SWE-Smith) and on a non-Qwen2.5 base (Qwen3-8B)"，或类似；正文里 4.2 节的 "Transfer across base-model families" 段也建议同步软化措辞。
- 我没自动改，是因为这是你想表达的论点强度问题。

### Q5. 摘要没写 SWE-Lite 的 gain
- 摘要现在只列了 Verified gain `+2.8/+3.0/+3.2`。SWE-Lite 上其实更亮眼（`+3.7/+4.0/+5.4`，Qwen3-8B 上 +5.4 是全场最大）。
- 建议：补一句 "and $+3.7/+4.0/+5.4$ on SWE-Lite"。
- 没自动加是因为摘要长度敏感，你斟酌。

### Q6. "doubles in parameters" 这一句是否要保留
- 文件：`sec/4_experiments.tex` "Consistent gains across model scales" 段。
- 改后只有 7B 和 14B 两个点（恰好 2×）。我把"$4{\times}$ in parameters"改成了"doubles in parameters"，技术上正确但 "consistent across scales" 的论据其实变薄了。
- 备选：**不再强调"scales"**，改成 "Consistent gains on the Qwen2.5-Coder series"，把跨家族证据让位给 Qwen3-8B 这一段，会更诚实一些。
- 这是你的论文叙事抉择，请定夺。

### Q7. 消融 Block (A) 的"no CoT 提升小 → 是因为预训练阶段已经见过"是你的解释
- 文件：`sec/4_experiments.tex` 现在写着大致："no CoT 仅 +0.13，归因于很多代码在预训练阶段已经见过；self-CoT 14.85 说明方法本身有效，不只靠蒸馏。"
- 这是按你给的方向写的。但严格说，这是一个**解释性主张**，没有直接实验支撑。
- 建议你过一下措辞，确认你愿意为这个 framing 背书；如果想更保守，可以改成 "consistent with the hypothesis that ..." 或类似。

### Q8. 摘要重写后的句式
- proofread agent 把摘要里"---a consistent gain across scales---and"改成了 "by $+2.8/+3.0$ points at 7B/14B and by a comparable $+3.2$ on Qwen3-8B"。
- 这只是消除别扭句式，没改数字。如果你不喜欢这个写法可以再调。

---

## 🟡 低优先级 — 体例/小问题

### Q9. 没被引用的 BibTeX 条目
`references.bib` 里下列条目被定义了但全文没有 \cite：
`starcoder2`, `qwen3`, `qwen3coder`, `mmlu`, `mathbench`, `ifeval`, `humanevalplus`

按你的指令我没自动删，但 NeurIPS 审稿一般会扫这种东西。建议要么补 cite 要么删条目。`qwen3` 和 `qwen3coder` 现在论文里有提到 Qwen3-8B，可能就忘了 \cite，建议尽快补上。

### Q10. 编译被本地环境卡住
- proofread agent 跑 `pdflatex neurips_2026.tex` 报 `File 'environ.sty' not found`，这是这台机器缺 `texlive-latex-extra`，**不是论文本身的问题**。
- 建议：你在自己常用的 Overleaf / 本地 TeX Live 环境编译一次确认能过。

### Q11. `tab:dataset` 与 `fig:dataset_categories` 没被 \ref
- 都在附录里有 `\label`，但正文/附录其它地方都没用 `\ref` 引到它们。
- 建议：在附录数据小节加一句 "(Table~\ref{tab:dataset}, Figure~\ref{fig:dataset_categories})"，否则审稿人会指出来。

### Q12. CSV 实际仓库行数与你给的 968 不一致（信息记录）
- 你给的文件叫 `code_repo_list_968.csv`，但里面实际有 988 行数据 + 1 行表头（去重后 987 个 unique repo URL）。
- 我**没**改你给的 968 这个数字 —— 严格按你"全部出现仓库数量990的地方都要改为968个"的指令执行，全文统一为 968。
- 如果你确认 CSV 应该被裁到 968 行（去掉重复或不合规仓库），需要你处理 CSV 本身或告诉我正确的统计口径。

---

## ✅ 已经完成、不用你管的事

- 全文 32B / Qwen2.5-Coder-32B 全部清除（abstract、intro、experiments 主表、analysis、conclusion、appendix）
- 主表 32B 那 5 行（含 \rowcolor）已删除；正文 "Consistent gains across model scales" 段重写为只覆盖 7B/14B
- mask 比例全部更新为 80% single + 15% pair + 5% triple；表格 (C) 行更新为 85%/15% 和 80/15/5
- 数据统计：2.6B token，320K/60K/20K 样本（2.0B/0.4B/0.2B token），78K 文件，平均 LoC/file=428，平均 target LoC=34
- 仓库数量 990 → 968（abstract、intro、method、appendix 多处）
- 方法部分加入 Gemini-3-Flash 验证段落："top $\sim$400K samples" + 引用 prompt 模板附录
- 删除了 "Full Qwen2.5-Coder Scaling Curves" 附录小节及其 \ref
- 方法流程图 `figs/pdg_score.pdf` 重新生成 —— 代码不再溢出、FIM 公式框收紧、图例展开、`class Calculator` 标签可见
- teaser 右子图改为 7B/14B/8B（Qwen3-8B 的真实数值），路径 `\includegraphics` 已切到 `figs/teaser_v34.pdf`，caption 已注明家族
- proofread agent 修复了三处一致性问题：BFCL/τ-bench 的小数差异、摘要句式、experiments 段的 4× 措辞
- `figs/plot_pdg_score.py` 与 `figs/teaser_final.py`（修改后的代码）都放在了输出目录的 `figs/` 下
