"""
Side-by-side report: PerturbEnergy's configurations against the published baselines.

    python scripts/report_compare.py --out compare.html

Reads eval.json from BOTH run roots. The two were produced by the same evaluation code on
the same splits -- verified by the reference predictors agreeing to 0.00e+00 (control) and
~1e-6 (mean_shift, a float32/float64 accumulation difference) -- so the columns are
directly comparable. Full Tahoe is excluded: only our side has it.

Same taxonomy as scripts/report.py: HVG split into (a) correlation, (b) DEG-related,
(c) distribution-level, then the DEG group as a K-sweep per metric family.
"""
import argparse
import glob
import json
import os

import _bootstrap  # noqa: F401
import numpy as np

from perturbenergy.evaluation.metrics import DEG_METRICS, HVG_METRICS

OURS_ROOT = "/data/projects/yyc/models/perturbenergy"
BASE_ROOT = "/data/projects/yyc/models/baselines"
DATASETS = ["norman", "replogle", "pbmc", "tahoe"]

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

REFS = [("ref_control", "control", "预测无变化"),
        ("ref_mean_shift", "mean_shift", "对照 + 训练集平均 delta（线性）"),
        ("ref_additive", "additive", "delta(A_B)=delta(A)+delta(B)（仅 Norman）")]
BASELINES = [("cpa", "CPA", "对抗解耦的条件自编码器"),
             ("scgen", "scGen", "潜空间向量算术"),
             ("cellot", "CellOT", "最优传输映射"),
             ("cellflow", "CellFlow", "流匹配"),
             ("state", "STATE", "集合到集合的 transformer"),
             ("perturbdiff", "PerturbDiff", "分布上的扩散"),
             ("squidiff", "Squidiff", "单细胞扩散"),
             ("stvae", "stVAE", "条件 VAE"),
             ("perturbedvae", "PerturbedVAE", "扰动条件 VAE")]
OURS = [("noenergy", "energy OFF", "同一 VAE，能量关闭（消融对照）"),
        ("energy_lang", "Langevin", "Langevin 负样本（论文 Eq.15）"),
        ("energy_mis", "mismatch", "错配扰动负样本"),
        ("energy_sel", "Lang+select", "Langevin，训练时能量参与 8 候选选择"),
        ("ft_lang", "2-stage", "先 VAE 后能量"),
        ("obs_pairing", "[abl] observed", "重建观测细胞本身（消融）")]


def load(ds):
    """-> [(label, kind, summary)] in display order."""
    out = []
    for group, root, kind in [(REFS, BASE_ROOT, "ref"),
                              (BASELINES, BASE_ROOT, "base"),
                              (OURS, OURS_ROOT, "ours")]:
        for suf, lab, _ in group:
            p = os.path.join(root, f"{ds}_{suf}", "eval.json")
            if os.path.exists(p):
                out.append((lab, kind, json.load(open(p))["summary"]))
    return out


def fnum(v, best, best_base, best_ours, hib):
    if not isinstance(v, (int, float)) or not np.isfinite(v):
        return '<td class="na">—</td>'
    cls = []
    if best is not None and abs(v - best) < 1e-12:
        cls.append("best")
    elif best_ours is not None and abs(v - best_ours) < 1e-12:
        cls.append("bestours")
    return (f'<td class="{" ".join(cls)}">{v:.4f}</td>' if cls
            else f"<td>{v:.4f}</td>")


def table(rows, cols, dirs, caption):
    cols = [c for c in cols if any(c in s for _, _, s in rows)]
    if not cols:
        return ""
    best, best_base, best_ours = {}, {}, {}
    for c in cols:
        def vals(kinds):
            return [s[c] for _, k, s in rows if k in kinds
                    and isinstance(s.get(c), (int, float)) and np.isfinite(s[c])]
        f = max if dirs[c] else min
        allv, bv, ov = vals({"ref", "base", "ours"}), vals({"base"}), vals({"ours"})
        if allv:
            best[c] = f(allv)
        if bv:
            best_base[c] = f(bv)
        if ov:
            best_ours[c] = f(ov)
    h = "".join(f'<th><span class="m">{c}</span>'
                f'<span class="dir">{"↑" if dirs[c] else "↓"}</span></th>' for c in cols)
    body, prev = [], None
    for lab, kind, s in rows:
        sep = ' class="sep"' if prev and prev != kind else ""
        prev = kind
        cells = "".join(fnum(s.get(c), best.get(c), best_base.get(c), best_ours.get(c),
                             dirs[c]) for c in cols)
        body.append(f'<tr{sep}><th scope="row"><span class="k {kind}"></span>{lab}</th>'
                    f"{cells}</tr>")
    return (f'<figure class="tw"><figcaption>{caption}</figcaption>'
            f'<div class="scroll"><table><thead><tr><th scope="col">方法</th>{h}</tr>'
            f"</thead><tbody>{''.join(body)}</tbody></table></div></figure>")


def scoreboard(rows, cols, dirs):
    """我们的最佳配置 vs 基线的最佳，逐指标数胜负。"""
    win = tie = loss = 0
    for c in cols:
        def vals(kinds):
            return [s[c] for _, k, s in rows if k in kinds
                    and isinstance(s.get(c), (int, float)) and np.isfinite(s[c])]
        bv, ov = vals({"base"}), vals({"ours"})
        if not bv or not ov:
            continue
        f = max if dirs[c] else min
        b, o = f(bv), f(ov)
        if abs(o - b) < 1e-12:
            tie += 1
        elif (o > b) == bool(dirs[c]):
            win += 1
        else:
            loss += 1
    return win, tie, loss


def build(out_path):
    order = [d for d in DATASETS if load(d)]
    tabs, panes, totals = [], [], {}
    for i, ds in enumerate(order):
        rows = load(ds)
        s0 = rows[0][2]
        n_base = sum(1 for _, k, _ in rows if k == "base")
        allcols, dirs = [], {}
        for _, _, cs in HVG_BLOCKS:
            for c in cs:
                if c in HVG_METRICS:
                    allcols.append(c); dirs[c] = HVG_METRICS[c][1]
        for fam in DEG_FAMILIES:
            for k in DE_KS:
                allcols.append(f"de{k}_{fam}"); dirs[f"de{k}_{fam}"] = DEG_METRICS[fam][0]
        w, t, l = scoreboard(rows, [c for c in allcols if any(c in s for _, _, s in rows)],
                             dirs)
        totals[ds] = (w, t, l, n_base)

        tabs.append(f'<button class="tab{" on" if i == 0 else ""}" data-ds="{ds}" '
                    f'role="tab" aria-selected="{"true" if i == 0 else "false"}">'
                    f'{ds.capitalize()}</button>')
        blocks = "".join(
            table(rows, cs, {c: HVG_METRICS[c][1] for c in cs}, f"<b>({k})</b> {name}")
            for k, name, cs in HVG_BLOCKS)
        degs = "".join(
            table(rows, [f"de{k}_{fam}" for k in DE_KS],
                  {f"de{k}_{fam}": DEG_METRICS[fam][0] for k in DE_KS},
                  f"<b>{fam}</b> over the true top-K DE genes")
            for fam in DEG_FAMILIES)
        panes.append(f"""
<section class="pane{' on' if i == 0 else ''}" data-ds="{ds}" role="tabpanel">
  <header class="dsh">
    <h2>{ds.capitalize()}</h2>
    <dl class="facts">
      <div><dt>划分</dt><dd>{s0.get('split_kind')}</dd></div>
      <div><dt>测试组</dt><dd>{s0.get('n_groups')}</dd></div>
      <div><dt>对比方法</dt><dd>{n_base} 个基线 + 我们的 {sum(1 for _,k,_ in rows if k=='ours')} 个配置</dd></div>
      <div><dt>逐指标胜负</dt><dd class="sb">{w} 胜 / {t} 平 / {l} 负</dd></div>
    </dl>
    <p class="note">「逐指标胜负」比较的是<b>我们最好的配置</b>与<b>基线中最好的</b>，
      在全部 HVG 与 DEG 指标上逐项计数。深色为全场最优，浅色为我们内部最优。</p>
  </header>
  <h3 class="gh">HVG 组 <span>全部 2,000 个基因</span></h3>
  {blocks}
  <h3 class="gh">DEG 组 <span>真实 top-K 差异表达基因</span></h3>
  {degs}
</section>""")

    legend = "".join(
        f'<div><dt><span class="k {kind}"></span>{lab}</dt><dd>{desc}</dd></div>'
        for group, kind in [(REFS, "ref"), (BASELINES, "base"), (OURS, "ours")]
        for _, lab, desc in group)
    HEAD = [("R2_delta", 1, "R²_delta"), ("PDCorr", 1, "PDCorr"),
            ("DEOverlap", 1, "DEOverlap"), ("de50_MSE", 0, "DEG50 MSE"),
            ("de25_R2_delta", 1, "DEG25 R²_delta")]
    cards = []
    for ds, (w, t, l, nb) in totals.items():
        rows = load(ds)
        cells = []
        for m, hib, lab in HEAD:
            def best(kind):
                v = [s_[m] for _, k, s_ in rows if k == kind
                     and isinstance(s_.get(m), (int, float)) and np.isfinite(s_[m])]
                return (max(v) if hib else min(v)) if v else None
            b, o = best("base"), best("ours")
            if b is None or o is None:
                continue
            better = (o > b) if hib else (o < b)
            cells.append(f'<div class="h2h{" win" if better else " lose"}">'
                         f'<span class="h2hm">{lab}</span>'
                         f'<span class="h2hv">{o:.4f}</span>'
                         f'<span class="h2hb">基线 {b:.4f}</span></div>')
        cards.append(f'<div class="sc"><div class="sch"><span class="scn">'
                     f'{ds.capitalize()}</span><span class="scd">逐指标 {w}/{t}/{l} '
                     f'（胜/平/负，共 {w+t+l} 项）</span></div>'
                     f'<div class="h2hs">{"".join(cells)}</div></div>')
    summary = "".join(cards)

    open(out_path, "w").write(f"""<title>扰动预测方法横评</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{{
  --bg:#F7F8FA; --surface:#FFFFFF; --ink:#16191F; --muted:#5B6472; --line:#E2E5EA;
  --accent:#0F766E; --accent-soft:#E4F1EF; --base:#8A6D3B; --base-soft:#F5EFE3;
  --warn:#B4443A; --shade:#FAFBFC;
  --sans:"Public Sans",-apple-system,BlinkMacSystemFont,sans-serif;
  --serif:"Newsreader",Georgia,serif;
  --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#101317; --surface:#171B21; --ink:#E6E9ED; --muted:#949CA8; --line:#262C34;
  --accent:#4FD1C5; --accent-soft:#12312E; --base:#C8A96A; --base-soft:#2A2318;
  --warn:#E07A6E; --shade:#13171C;
}}}}
:root[data-theme="dark"]{{
  --bg:#101317; --surface:#171B21; --ink:#E6E9ED; --muted:#949CA8; --line:#262C34;
  --accent:#4FD1C5; --accent-soft:#12312E; --base:#C8A96A; --base-soft:#2A2318;
  --warn:#E07A6E; --shade:#13171C;
}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;
  line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1320px;margin:0 auto;padding:40px 28px 80px;display:flex;
  flex-direction:column;gap:26px}}
h1{{font-family:var(--serif);font-weight:600;font-size:2.1rem;line-height:1.2;margin:0;
  letter-spacing:-.01em;text-wrap:balance}}
.lede{{color:var(--muted);max-width:66ch;margin:0}}
.rule{{height:1px;background:var(--line);border:0;margin:0}}
.scores{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}
.sc{{background:var(--surface);border:1px solid var(--line);border-radius:6px;
  padding:13px 15px;display:flex;flex-direction:column;gap:9px}}
.sch{{display:flex;flex-direction:column;gap:1px}}
.h2hs{{display:flex;flex-direction:column;gap:5px}}
.h2h{{display:grid;grid-template-columns:1fr auto;gap:0 8px;align-items:baseline;
  padding:4px 8px;border-radius:4px;background:var(--shade);
  border-left:2px solid var(--muted)}}
.h2h.win{{border-left-color:var(--accent)}}
.h2h.lose{{border-left-color:var(--base)}}
.h2hm{{font-family:var(--mono);font-size:.68rem;color:var(--muted)}}
.h2hv{{font-weight:600;font-variant-numeric:tabular-nums;text-align:right}}
.h2h.win .h2hv{{color:var(--accent)}}
.h2h.lose .h2hv{{color:var(--base)}}
.h2hb{{grid-column:1/-1;font-size:.7rem;color:var(--muted);
  font-variant-numeric:tabular-nums}}
.caveat{{margin:0;color:var(--muted);font-size:.86rem;max-width:80ch;
  border-left:2px solid var(--warn);padding-left:13px}}
.caveat b{{color:var(--ink);font-weight:600}}
.scn{{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;
  letter-spacing:.09em;color:var(--muted)}}
.scd{{font-size:.72rem;color:var(--muted)}}
.legend{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
  gap:1px 26px;margin:0;font-size:.85rem}}
.legend>div{{display:flex;gap:9px;align-items:baseline;padding:2px 0}}
.legend dt{{font-weight:600;white-space:nowrap;display:flex;align-items:center;gap:7px;
  min-width:110px}}
.legend dd{{margin:0;color:var(--muted)}}
.k{{width:8px;height:8px;border-radius:2px;flex:none;display:inline-block}}
.k.ref{{background:var(--muted)}}
.k.base{{background:var(--base)}}
.k.ours{{background:var(--accent)}}
.tabs{{display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--line)}}
.tab{{font:inherit;font-weight:500;color:var(--muted);background:none;border:0;
  border-bottom:2px solid transparent;padding:9px 16px;cursor:pointer;margin-bottom:-1px}}
.tab:hover{{color:var(--ink)}}
.tab.on{{color:var(--accent);border-bottom-color:var(--accent)}}
.tab:focus-visible{{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}}
.pane{{display:none;flex-direction:column;gap:20px}}
.pane.on{{display:flex}}
.dsh{{display:flex;flex-direction:column;gap:11px}}
.dsh h2{{font-family:var(--serif);font-weight:600;font-size:1.5rem;margin:0}}
.facts{{display:flex;flex-wrap:wrap;gap:0 32px;margin:0}}
.facts div{{display:flex;flex-direction:column;gap:1px}}
.facts dt{{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;
  letter-spacing:.09em;color:var(--muted)}}
.facts dd{{margin:0;font-weight:500;font-variant-numeric:tabular-nums}}
.facts dd.sb{{color:var(--accent);font-weight:600}}
.note{{margin:0;color:var(--muted);max-width:78ch;font-size:.88rem;
  border-left:2px solid var(--line);padding-left:13px}}
.note b{{color:var(--ink);font-weight:600}}
.gh{{font-family:var(--serif);font-weight:500;font-size:1.15rem;margin:10px 0 0;
  display:flex;align-items:baseline;gap:11px;border-top:1px solid var(--line);
  padding-top:17px}}
.gh span{{font-family:var(--mono);font-size:.7rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.08em}}
.tw{{margin:0;display:flex;flex-direction:column;gap:6px}}
figcaption{{font-size:.85rem;color:var(--muted)}}
figcaption b{{color:var(--ink);font-family:var(--mono);font-weight:500}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:5px;
  background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:.82rem;
  font-variant-numeric:tabular-nums}}
thead th{{position:sticky;top:0;background:var(--surface);text-align:right;
  padding:9px 11px;border-bottom:1px solid var(--line);white-space:nowrap;
  font-weight:500;z-index:1}}
thead th:first-child{{text-align:left}}
.m{{font-family:var(--mono);font-size:.75rem}}
.dir{{color:var(--muted);margin-left:5px;font-size:.72rem}}
tbody th{{text-align:left;font-weight:500;white-space:nowrap;padding:7px 11px;
  display:flex;align-items:center;gap:8px}}
tbody td{{text-align:right;padding:7px 11px;color:var(--muted)}}
tbody tr:nth-child(even){{background:var(--shade)}}
tbody tr:hover td,tbody tr:hover th{{background:var(--accent-soft)}}
tr.sep th,tr.sep td{{border-top:2px solid var(--line)}}
td.best{{background:var(--accent-soft);color:var(--accent);font-weight:600}}
td.bestours{{color:var(--accent)}}
td.na{{color:var(--line)}}
footer{{color:var(--muted);font-size:.82rem;border-top:1px solid var(--line);
  padding-top:17px}}
@media (max-width:640px){{.wrap{{padding:26px 15px 60px}}h1{{font-size:1.6rem}}}}
</style>
<div class="wrap">
  <header>
    <h1>扰动预测方法横评</h1>
    <p class="lede">九个已发表方法、三个参考预测器，与 PerturbEnergy 的六种配置，
    在四个数据集上用同一套协议评测。两侧结果由同一份评估代码在同一划分上产生 —
    参考预测器 <code>control</code> 双方数值完全相同（0.00e+00），可直接并表比较。</p>
  </header>
  <hr class="rule">
  <div class="scores">{summary}</div>
  <p class="caveat">上面每格给出<b>我们最好的配置</b>与<b>基线中最好的</b>的直接数值，
    青色表示我们更优。旁边的「逐指标胜/平/负」覆盖全部 59 个指标，但请谨慎解读：
    其中许多高度相关（PCC/R²/PDCorr/LogFC_cos 基本测同一件事，DEG 组 8 个指标族 × 4 个 K
    更是如此），因此一致的<b>小幅</b>差距会被计数放大成悬殊比分。
    应以实际数值差距为准。</p>
  <dl class="legend">{legend}</dl>
  <div class="tabs" role="tablist">{''.join(tabs)}</div>
  {''.join(panes)}
  <footer>DE 基因始终在真实数据上排序（perturbed vs 匹配对照，Wilcoxon 秩和），
  所有方法在同一组基因上评分。PDS 报的是归一化排名（越低越好，随机≈0.5）。
  全量 Tahoe-100M 不在此页 — 基线侧没有对应版本。</footer>
</div>
<script>
const tabs=[...document.querySelectorAll('.tab')],panes=[...document.querySelectorAll('.pane')];
function show(ds){{
  tabs.forEach(t=>{{const on=t.dataset.ds===ds;t.classList.toggle('on',on);
    t.setAttribute('aria-selected',on)}});
  panes.forEach(p=>p.classList.toggle('on',p.dataset.ds===ds));
  try{{localStorage.setItem('pe-cmp-ds',ds)}}catch(e){{}}
}}
tabs.forEach(t=>t.addEventListener('click',()=>show(t.dataset.ds)));
try{{const s=localStorage.getItem('pe-cmp-ds');
  if(s&&tabs.some(t=>t.dataset.ds===s))show(s)}}catch(e){{}}
</script>""")
    return totals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = build(a.out)
    print(f"wrote {a.out}")
    for ds, (w, ti, l, nb) in t.items():
        print(f"  {ds:9s} {w} 胜 / {ti} 平 / {l} 负  (对 {nb} 个基线)")
