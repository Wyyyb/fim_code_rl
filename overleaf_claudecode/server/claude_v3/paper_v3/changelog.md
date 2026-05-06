# paper_v3 修订日志 (changelog)

输出目录：`/data/yubowang/fim_code_rl/overleaf_claudecode/claude_v3/paper_v3/`

主协调 agent 调度了 4 个子 agent（按时间顺序）：

| Sub-agent | 子任务编号 | 内容 | 修改文件 |
| --- | --- | --- | --- |
| **text-edits agent**（general-purpose） | 1, 2, 3, 4, 5, 6 | 全部 .tex 文本改动（消融重写、mask 比例 80/15/5、数据统计 2.6B/968、删除全部 32B、加入 Gemini-3-Flash 验证段落、删除 Full Qwen2.5-Coder Scaling Curves 附录小节） | `sec/1_abstract.tex`, `sec/2_introduction.tex`, `sec/3_method.tex`, `sec/4_experiments.tex`, `sec/8_conclusion.tex`, `sec/9_appendix.tex` |
| **method-figure agent**（general-purpose） | 7 | 重新生成方法流程图：左侧代码框不再溢出；FIM 公式框收紧不超边界；图例 `E_call / E_sib / top-level / in-class` 重新分布不重叠；`class Calculator` 标签上移到绿色虚线框上方完全可见；并迭代渲染验证 | `figs/plot_pdg_score.py`, `figs/pdg_score.pdf`, `figs/pdg_score.png` |
| **teaser-figure agent**（general-purpose） | 8 | 右子图改为 7B / 14B / 8B（去掉 32B），把 8B 数据替换为 Qwen3-8B 实际值（V: 31.8/35.0；L: 27.3/32.7），标题改为 "SWE-Bench gains across models"，并迭代渲染验证；同步更新 caption 注明 7B/14B 是 Qwen2.5-Coder、8B 是 Qwen3；并把 `\includegraphics` 路径改为 `figs/teaser_v34.pdf` | `figs/teaser_final.py`, `figs/teaser_v34.pdf`, `figs/teaser_v34.png`, `sec/2_introduction.tex` |
| **proofread agent**（general-purpose） | 9 | 通读全文，校对一致性（数值、引用、标签）；机械性修复见下表；同时把需要作者决策的问题写入 `REVIEW_FINDINGS.md`（英文） | `sec/1_abstract.tex`, `sec/2_introduction.tex`, `sec/4_experiments.tex`, `REVIEW_FINDINGS.md` |

## proofread agent 机械性修复

| 文件 | 改动 |
| --- | --- |
| `sec/2_introduction.tex` | "$+2.5$ on BFCL / $+4.0$ on $\tau$-bench" → "$+2.4$ / $+3.9$"，与 `tab:cap` 对齐 |
| `sec/1_abstract.tex` | 重写 "---a consistent gain across scales---and" 处的别扭句式，改为 "by $+2.8/+3.0$ points at 7B/14B and by a comparable $+3.2$ on Qwen3-8B" |
| `sec/4_experiments.tex` | "stays in a narrow band as the base model grows by $4{\times}$ in parameters" → "is essentially unchanged as the base model doubles in parameters"，并把外推措辞 hedged 为 "in the practical deployment range we evaluate" |

## 协调步骤（主 agent 行为记录）

1. 复制 `paper_v2/` 到 `paper_v3/` 作为工作副本。
2. 把 `teaser_plot/teaser_final.py` 复制到 `paper_v3/figs/teaser_final.py`，让 teaser 子 agent 在输出目录内修改。
3. **并行**派发 method-figure 与 teaser-figure 两个图相关 agent（互不依赖），同时派发 text-edits agent；3 个 agent 同时工作。
4. 两个图 agent 完成且文本编辑完成后，串行派发 proofread agent。
5. 收尾：清理临时文件（`plot_pdg_score.py.bak`、`neurips_2026.log`），写入 `changelog.md`（本文件）和 `questions.md`（中文，作者待确认事项）。

## 输出目录主要产物

- `sec/*.tex`、`neurips_2026.tex`、`references.bib`：修改后的论文源
- `figs/plot_pdg_score.py`：修改后的方法图代码
- `figs/teaser_final.py`：修改后的 teaser 代码
- `figs/pdg_score.pdf` / `figs/teaser_v34.pdf`：新版图像（已嵌入论文）
- `REVIEW_FINDINGS.md`：proofread agent 写的英文 findings 报告
- `questions.md`：中文版作者待办（请优先看这个）
- `changelog.md`：本文件
