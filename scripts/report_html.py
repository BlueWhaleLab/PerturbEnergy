"""
Render every eval.json under the run root into one standalone HTML report.

    python scripts/report_html.py --out report.html

Same taxonomy as scripts/report.py -- HVG split into (a) correlation, (b) DEG-related,
(c) distribution-level, then the DEG group as a K-sweep per metric family. Re-run it to
refresh the page after more evaluations land.
"""
import argparse
import glob
import json
import os

import _bootstrap  # noqa: F401
import numpy as np

from perturbenergy import paths as P
from perturbenergy.evaluation.metrics import DEG_METRICS, HVG_METRICS

HVG_BLOCKS = [
    ("a", "correlation-based measures",
     ["PCC", "R2", "PDCorr", "R2_delta", "LFCSpear", "PCA_LFCSpear", "LogFC_cos",
      "DirAgr", "RLogFC", "PDS_cos", "PDS_l1", "PDS_l2"]),
    ("b", "DEG-related scores",
     ["DEPrec", "DERec", "DEOverlap", "ES", "AUROC", "AUPRC"]),
    ("c", "distribution-level distances",
     ["RMSE", "MSE", "MAE", "RMSE_cell", "R2_cell", "EDist", "MMD", "Sinkhorn",
      "PCA_Sinkhorn"]),
]
DEG_FAMILIES = ["PCC", "R2", "R2_delta", "MAE", "MSE", "RMSE", "RMSE_cell", "R2_cell"]
DE_KS = [25, 50, 100, 200]

VARIANTS = [
    ("ref_control", "control", "baseline", "预测无变化：所有 delta 类指标的零点"),
    ("ref_mean_shift", "mean_shift", "baseline", "对照 + 该扰动在训练集上的平均 delta"),
    ("ref_additive", "additive", "baseline", "delta(A_B)=delta(A)+delta(B)，无交互假设"),
    ("noenergy", "energy OFF", "model", "同一个 VAE，能量项关闭 — 消融对照"),
    ("energy_lang", "Langevin", "model", "联合训练，负样本来自 Langevin 链（Eq. 15）"),
    ("energy_mis", "mismatch", "model", "联合训练，负样本 = z_a 配错配扰动的 embedding"),
    ("energy_sel", "Langevin + select", "model", "同 Langevin，训练时能量也参与候选选择 (8 候选)"),
    ("ft_lang", "2-stage finetune", "model", "先纯 VAE，再开能量"),
    ("obs_pairing", "observed pairing", "ablation", "重建观测细胞本身（Eq. 7 字面读法）"),
]

DATASET_NOTES = {
    "norman": ("Norman2019 · CRISPRa", "111,445 cells · 全量",
               "106 单靶点 + 131 配对靶点，单一 K562 背景。留出 30% 的配对扰动，"
               "其两个靶点仍在训练集中出现 — 组合泛化。PerturbDiff 未使用此数据集。", False),
    "replogle": ("Replogle · CRISPRi", "643,413 cells · 全量",
                 "留出细胞系 hepg2，30% 扰动回流训练。与 PerturbDiff 同一份文件、"
                 "同一协议、同一规模。", False),
    "pbmc": ("PBMC · 细胞因子", "1,674,722 cells · 原始的 17.3%",
             "留出 4/12 供体，30% 扰动回流。测试条件 63 个（论文 62）— 条件全覆盖，"
             "但每组细胞 cap 100，训练数据量为论文的 1/5.8。", True),
    "tahoe": ("Tahoe-100M · 药物", "1,922,974 cells · 原始的 1.9%",
              "PCA 选 5/50 细胞系留出，30% 回流。<b>本栏的扰动单位是「药物」(380 个)，"
              "而非论文的「药物×剂量」(1,137 个)</b>：Tahoe 有 379 种药 + DMSO，但 1,344 个 "
              "(drug, sample) 组合，每种药约 3 个剂量。把剂量混为一个条件并非无害 — 实测同药"
              "不同剂量之间 delta 的余弦相似度中位仅 <b>0.389</b>（38% 的组低于 0.3），最强/最弱"
              "剂量的模长比中位 1.6×，即不同剂量触发的是部分不同的转录程序。因此本栏的真实 delta "
              "不对应任何真实实验条件，DE 基因集合被稀释，delta 类指标被<b>低估</b>；同时判别池只有 "
              "380 而非 1,317，PDS 类指标被<b>高估</b>。能量 vs 无能量的相对排序仍可信（同一定义下"
              "比较），但绝对值不宜引用。全量正确版本（95.6M cells，约 1,317 个扰动）正在准备。",
              True),
}


def collect(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "eval*.json"), recursive=True)):
        try:
            s = json.load(open(f))["summary"]
        except Exception:                                              # noqa: BLE001
            continue
        s["_run"] = os.path.basename(os.path.dirname(f))
        out.setdefault(s.get("dataset", "?"), {})[s["_run"]] = s
    return out


def fnum(v, hib, best):
    if not isinstance(v, (int, float)) or not np.isfinite(v):
        return '<td class="na">—</td>'
    cls = []
    if best is not None and abs(v - best) < 1e-12:
        cls.append("best")
    if v < 0 and abs(v) > 1e-9:
        cls.append("neg")
    c = f' class="{" ".join(cls)}"' if cls else ""
    return f"<td{c}>{v:.4f}</td>"


def table(rows, cols, dirs, caption):
    cols = [c for c in cols if any(c in r for _, _, r in rows)]
    if not cols:
        return ""
    best = {}
    for c in cols:
        v = [r[c] for _, _, r in rows
             if isinstance(r.get(c), (int, float)) and np.isfinite(r[c])]
        if v:
            best[c] = max(v) if dirs[c] else min(v)
    h = "".join(f'<th><span class="m">{c}</span>'
                f'<span class="dir">{"↑" if dirs[c] else "↓"}</span></th>' for c in cols)
    body = []
    prev = None
    for label, kind, r in rows:
        sep = ' class="sep"' if prev and prev != kind else ""
        prev = kind
        cells = "".join(fnum(r.get(c), dirs[c], best.get(c)) for c in cols)
        body.append(f'<tr{sep}><th scope="row"><span class="k {kind}"></span>{label}</th>'
                    f"{cells}</tr>")
    return (f'<figure class="tw"><figcaption>{caption}</figcaption>'
            f'<div class="scroll"><table><thead><tr><th scope="col">setting</th>{h}</tr>'
            f"</thead><tbody>{''.join(body)}</tbody></table></div></figure>")


def build(by_ds):
    order = [d for d in ("norman", "replogle", "pbmc", "tahoe") if d in by_ds]
    tabs, panes = [], []
    for i, ds in enumerate(order):
        runs = by_ds[ds]
        rows = [(lbl, kind, runs[f"{ds}_{suf}"])
                for suf, lbl, kind, _ in VARIANTS if f"{ds}_{suf}" in runs]
        if not rows:
            continue
        r0 = rows[0][2]
        title, scale, note, warn = DATASET_NOTES.get(ds, (ds, "", "", False))
        tabs.append(f'<button class="tab{" on" if i == 0 else ""}" data-ds="{ds}" '
                    f'role="tab" aria-selected="{"true" if i == 0 else "false"}">'
                    f'{title.split(" · ")[0]}</button>')

        blocks = "".join(
            table(rows, cols, {c: HVG_METRICS[c][1] for c in cols},
                  f'<b>({k})</b> {name}')
            for k, name, cols in HVG_BLOCKS)

        # every setting must carry the full K sweep, otherwise a half-filled table
        # would read as the final answer
        have = sum(f"de{DE_KS[0]}_PCC" in r for _, _, r in rows)
        if have == len(rows):
            degs = "".join(
                table(rows, [f"de{k}_{fam}" for k in DE_KS],
                      {f"de{k}_{fam}": DEG_METRICS[fam][0] for k in DE_KS},
                      f"<b>{fam}</b> over the true top-K DE genes")
                for fam in DEG_FAMILIES)
        else:
            degs = (f'<p class="pending">DEG 组 (K = 25 / 50 / 100 / 200) 评估进行中：'
                    f'{have}/{len(rows)} 个设定已完成。全部跑完后此页更新。</p>')

        panes.append(f"""
<section class="pane{' on' if i == 0 else ''}" data-ds="{ds}" role="tabpanel">
  <header class="dsh">
    <h2>{title}</h2>
    <dl class="facts">
      <div><dt>规模</dt><dd>{scale}</dd></div>
      <div><dt>划分</dt><dd>{r0.get('split_kind')}</dd></div>
      <div><dt>测试组</dt><dd>{r0.get('n_groups')}</dd></div>
      <div><dt>DE 排序</dt><dd>{r0.get('de_method')} · 真实数据</dd></div>
    </dl>
    <p class="note{' warn' if warn else ''}">{note}</p>
  </header>
  <h3 class="gh">HVG 组 <span>全部 2,000 个基因</span></h3>
  {blocks}
  <h3 class="gh">DEG 组 <span>真实 top-K 差异表达基因</span></h3>
  {degs}
</section>""")

    legend = "".join(
        f'<div><dt><span class="k {kind}"></span>{lbl}</dt><dd>{desc}</dd></div>'
        for _, lbl, kind, desc in VARIANTS)

    return f"""<title>PerturbEnergy 基准结果</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{{
  --bg:#F7F8FA; --surface:#FFFFFF; --ink:#16191F; --muted:#5B6472; --line:#E2E5EA;
  --accent:#0F766E; --accent-soft:#E4F1EF; --warn:#B4443A; --shade:#FAFBFC;
  --sans:"Public Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --serif:"Newsreader",Georgia,serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#101317; --surface:#171B21; --ink:#E6E9ED; --muted:#949CA8; --line:#262C34;
  --accent:#4FD1C5; --accent-soft:#12312E; --warn:#E07A6E; --shade:#13171C;
}}}}
:root[data-theme="dark"]{{
  --bg:#101317; --surface:#171B21; --ink:#E6E9ED; --muted:#949CA8; --line:#262C34;
  --accent:#4FD1C5; --accent-soft:#12312E; --warn:#E07A6E; --shade:#13171C;
}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1320px;margin:0 auto;padding:40px 28px 80px;
  display:flex;flex-direction:column;gap:28px}}
h1{{font-family:var(--serif);font-weight:600;font-size:2.1rem;line-height:1.2;
  letter-spacing:-.01em;margin:0;text-wrap:balance}}
.lede{{color:var(--muted);max-width:64ch;margin:0}}
.rule{{height:1px;background:var(--line);border:0;margin:0}}

.legend{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));
  gap:2px 26px;margin:0;font-size:.86rem}}
.legend>div{{display:flex;gap:10px;align-items:baseline;padding:3px 0}}
.legend dt{{font-weight:600;white-space:nowrap;display:flex;align-items:center;gap:7px}}
.legend dd{{margin:0;color:var(--muted)}}
.k{{width:8px;height:8px;border-radius:2px;flex:none;display:inline-block}}
.k.baseline{{background:var(--muted)}}
.k.model{{background:var(--accent)}}
.k.ablation{{background:transparent;border:1px solid var(--muted)}}

.tabs{{display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--line)}}
.tab{{font:inherit;font-weight:500;color:var(--muted);background:none;border:0;
  border-bottom:2px solid transparent;padding:9px 15px;cursor:pointer;margin-bottom:-1px}}
.tab:hover{{color:var(--ink)}}
.tab.on{{color:var(--accent);border-bottom-color:var(--accent)}}
.tab:focus-visible{{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}}

.pane{{display:none;flex-direction:column;gap:22px}}
.pane.on{{display:flex}}
.dsh{{display:flex;flex-direction:column;gap:12px}}
.dsh h2{{font-family:var(--serif);font-weight:600;font-size:1.5rem;margin:0}}
.facts{{display:flex;flex-wrap:wrap;gap:0 34px;margin:0}}
.facts div{{display:flex;flex-direction:column;gap:1px}}
.facts dt{{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;
  letter-spacing:.09em;color:var(--muted)}}
.facts dd{{margin:0;font-weight:500;font-variant-numeric:tabular-nums}}
.note{{margin:0;color:var(--muted);max-width:78ch;font-size:.9rem;
  border-left:2px solid var(--line);padding-left:13px}}
.note.warn{{border-left-color:var(--warn);color:var(--ink)}}
.note b{{color:var(--ink);font-weight:600}}
.note.warn b{{color:var(--warn)}}

.gh{{font-family:var(--serif);font-weight:500;font-size:1.15rem;margin:12px 0 0;
  display:flex;align-items:baseline;gap:11px;
  border-top:1px solid var(--line);padding-top:18px}}
.gh span{{font-family:var(--mono);font-size:.7rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.08em}}

.tw{{margin:0;display:flex;flex-direction:column;gap:7px}}
figcaption{{font-size:.85rem;color:var(--muted)}}
figcaption b{{color:var(--ink);font-family:var(--mono);font-weight:500}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:5px;
  background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:.83rem;
  font-variant-numeric:tabular-nums}}
thead th{{position:sticky;top:0;background:var(--surface);text-align:right;
  padding:9px 11px;border-bottom:1px solid var(--line);white-space:nowrap;
  font-weight:500;z-index:1}}
thead th:first-child{{text-align:left}}
.m{{font-family:var(--mono);font-size:.76rem}}
.dir{{color:var(--muted);margin-left:5px;font-size:.72rem}}
tbody th{{text-align:left;font-weight:500;white-space:nowrap;padding:7px 11px;
  display:flex;align-items:center;gap:8px}}
tbody td{{text-align:right;padding:7px 11px;color:var(--muted)}}
tbody tr:nth-child(even){{background:var(--shade)}}
tbody tr:hover td,tbody tr:hover th{{background:var(--accent-soft)}}
tr.sep th,tr.sep td{{border-top:1px solid var(--line)}}
td.best{{background:var(--accent-soft);color:var(--accent);font-weight:600}}
td.neg{{color:var(--warn)}}
td.na{{color:var(--line)}}
.pending{{color:var(--muted);font-style:italic;margin:0;padding:16px;
  border:1px dashed var(--line);border-radius:5px}}
footer{{color:var(--muted);font-size:.82rem;border-top:1px solid var(--line);
  padding-top:18px}}
@media (max-width:640px){{.wrap{{padding:26px 15px 60px}}h1{{font-size:1.6rem}}}}
</style>
<div class="wrap">
  <header>
    <h1>PerturbEnergy 基准结果</h1>
    <p class="lede">条件能量先验 p<sub>α</sub>(z<sub>a</sub>|z<sub>b</sub>,a) 对细胞扰动响应预测的影响。
    四个数据集，每个下列所有设定共享同一份数据、同样的训练步数与随机种子，唯一变量是能量项。
    指标分类沿用 <i>Benchmarking virtual cell models for in-the-wild perturbation response</i>
    的体系。每列最优值以青色标出；<span style="color:var(--warn)">红色</span>表示负值 —— 
    对 R²<sub>delta</sub> 而言即「比什么都不做还差」。</p>
  </header>
  <hr class="rule">
  <dl class="legend">{legend}</dl>
  <div class="tabs" role="tablist">{''.join(tabs)}</div>
  {''.join(panes)}
  <footer>DE 基因始终在真实数据上排序（perturbed vs 匹配对照，Wilcoxon 秩和），
  因此所有方法在同一组基因上被评分。PDS 报的是归一化排名（越低越好，随机≈0.5）；
  PerturbDiff 的口径是 1−该值，对照论文数字时需取补。</footer>
</div>
<script>
const tabs=[...document.querySelectorAll('.tab')],panes=[...document.querySelectorAll('.pane')];
function show(ds){{
  tabs.forEach(t=>{{const on=t.dataset.ds===ds;t.classList.toggle('on',on);
    t.setAttribute('aria-selected',on)}});
  panes.forEach(p=>p.classList.toggle('on',p.dataset.ds===ds));
  try{{localStorage.setItem('pe-ds',ds)}}catch(e){{}}
}}
tabs.forEach(t=>t.addEventListener('click',()=>show(t.dataset.ds)));
try{{const s=localStorage.getItem('pe-ds');
  if(s&&tabs.some(t=>t.dataset.ds===s))show(s)}}catch(e){{}}
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=P.MODELS_ROOT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_ds = collect(a.root)
    if not by_ds:
        raise SystemExit(f"no eval.json under {a.root}")
    open(a.out, "w").write(build(by_ds))
    n = sum(len(v) for v in by_ds.values())
    print(f"wrote {a.out}  ({len(by_ds)} datasets, {n} evaluations)")


if __name__ == "__main__":
    main()
