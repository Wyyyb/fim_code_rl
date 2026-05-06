# 待你处理的问题清单（Round 2）

输出目录：`/data/yubowang/fim_code_rl/overleaf_claudecode/claude_v3/paper_v3/`

下面是 round 2 改动后仍需要你过目或补数据的事项。每条都标了优先级。
**Round 1 中已经在 round 2 顺手解决的问题不再列**（详见下方"已解决"小节）。

---

## 🔴 高优先级 — 需要补数据

### Q1. License 表逐项数量与百分比
- 文件：`sec/9_appendix.tex` 中 `tab:licenses`，以及表前一段 prose。
- 现状：表里每个 license 的 count 仍是 `\textcolor{red}{TBD}`，prose 里以前那句"$\sim$79\% permissive / $\sim$5\% copyleft / $\sim$2\% Creative Commons"也已替换为 `\textcolor{red}{TBD}`（因为 968 仓库的统计你还没给）。
- 待办：填入 968 仓库新算的逐项 count + 三个百分比。
- 我**没自动从 csv 算**，因为：你给的 `code_repo_list_968.csv` 实际有 988 行（去重后 987 个 unique repo URL）但文件名是 968 —— CSV 与目标行数不一致，需要你确认正确的统计口径再填。

### Q2. 主表 std 占位
- 文件：`sec/4_experiments.tex` `tab:main`。
- 状态：原先 setup 段的 "（红色 ±? 是占位）"那句已删除。表里 R2E-Gym (officially reported) 的 7B/14B 行 std 看上去仍可能是估算（`$\pm$1.0/$\pm$0.8/$\pm$1.4/$\pm$0.7` 等）。
- 待办：确认或替换为最终复现实验的真实 std。

---

## 🟠 中优先级 — 体例 / 内容选择

### Q3. SWE-Lego 仓库的 license 与 cite
- 现 `references.bib` 里 `swelego` 是占位 misc 条目，你检查一下作者/年份/链接信息是否完整（round 1 之前就有，没改）。

### Q4. Hyperparameter 表里 global batch size 的 GPU 数假设
- r2-hyperparameter agent 假设了 `8 GPUs` 来计算 effective batch size（mid-train global bs=128，R2E-Gym=8，SWE-Smith=32，SWE-Lego=64）。
- 待办：如果你实际跑的不是 8 卡，告诉我真实数字我修正；或者如果你想保持抽象，我可以把 global bs 一列改成 `per_device_bs × grad_accum × N\,GPUs`。

### Q5. SWE-Smith torchtune 的 sharding 策略
- agent 写成了 "FSDP"（torchtune full-finetune-distributed 默认值），但 YAML 里没明确暴露 sharding key。如果实际使用了别的，告诉我。

### Q6. R2E-Gym weight decay
- YAML 没有 `weight_decay` 字段，agent 填了 `0.0 (default)`（HF Trainer / LlamaFactory 默认）。如果你实际跑时设了别的值，告诉我。

### Q7. 摘要修订后是否再润一遍
- Round 2 摘要已加 SWE-Lite gain `+3.7/+4.0/+5.4`、软化跨家族说法、补了 SWE-Lego 进 pipeline 列表。但摘要现在大概率比原来长一点点。你过一眼字数和节奏。

### Q8. checklist Q10 (Broader impacts) 与 Q8 (Compute resources) 答的是 No
- Broader impacts: 我们没单独章节；agent 答 No 并附简短说明。如果你认为 limitations 里那一两句可以算 partial 覆盖，可以改成 Yes。
- Compute resources: 现在没汇报 GPU-hours，agent 答 No。如果你能补到 appendix（"训练用了 N 张 H100 共 M 小时"），就可以改成 Yes。

### Q9. 篇幅
- 你说目标 9.5 页，现在编译我没办法直接验证（这台机器缺 `texlive-latex-extra`）。r2-text-edits agent 已经做了 copy-edit（去除冗余形容词/连接词），但具体落到几页要等你在自己的环境编译看一下。
- 如果还溢出，常用的下一步压缩是：appendix 里 hyperparameter 两张表合并成一张宽表；或者把 Limitations 收紧一段。

---

## 🟡 低优先级

### Q10. `tab:dataset` / `fig:dataset_categories` 仍未被 \ref
- Round 1 提过的小问题，r2 没改。建议在 appendix 附数据小节加 `(Table~\ref{tab:dataset}, Figure~\ref{fig:dataset_categories})` 让审稿人不挑。

### Q11. 编译验证
- 本机仍缺 `environ.sty`，没法 pdflatex 完整编译。请在你常用环境编译一次确认全篇通过、表格不出框、引用都能对上。

---

## ✅ Round 2 已完成（不用你管）

- 模型命名统一：`Qwen2.5-Coder-7B-Instruct`、`Qwen2.5-Coder-14B-Instruct`、`Qwen3-8B (base, not Instruct)`
- Teaser caption 中段简化为 "We mid-train the model to fill in $B$'s body together with a chain-of-thought rationale, given the surrounding file as an FIM-formatted prompt."
- Method "Motivation" 段精简（去重四阶段表述合并到一处，~25% 字数下降）
- 方法图所有问题修复：`mean→total` 箭杆完整可见、`class Calculator` 完全在框内、FIM 公式不再出框（靠收紧 wspace 让 panel b 拓宽）
- 引用补全 + 名字大小写统一：`qwen3` 已 cite；`R2E-Gym` / `SWE-Lego` / `SWE-Smith` 全篇大小写一致；删除未用 bib 条目 `starcoder2 / qwen3coder / mmlu / mathbench / ifeval / humanevalplus`
- Appendix Hyperparameters stub 替换为两张完整表（mid-train 单列 + post-train 三列），含框架、lr、warmup、wd、epochs、bs、seq、system 信息，SWE-Lego 2-epoch 偏离用脚注说明
- NeurIPS checklist 全部填完（16 项 + justification + 章节引用），`answerTODO` / `justificationTODO` 均为 0
- 通篇删除 placeholder / stub / "awaiting final" 等占位语；只保留确实缺数据处的 `\textcolor{red}{TBD}`
- 摘要：SWE-Lite gain 加入、跨家族说法软化、SWE-Lego 加入 pipeline 列表
- 4.2 段标题 "Consistent gains across model scales" 改为 "Consistent gains on the Qwen2.5-Coder series"
- 消融 no-CoT 解释改成 hypothesis 措辞（"a small effect consistent with the hypothesis that..."）
- 字数级 copy-edit（去重多余形容词/连接词），等你编译后看具体到几页
