import sys
import json

import numpy as np
import pandas as pd
import os.path as osp

import matplotlib.pyplot as plt

sys.path.append(osp.dirname(osp.dirname(__file__)))
from scipy import stats
from tqdm import tqdm
from verl_tool.workers.reward_manager.search_r1_qa_em import compute_score

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy.stats import mannwhitneyu, spearmanr, gaussian_kde

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "figure.autolayout": False,   # we’ll call tight_layout() explicitly
})

def _aggregate_y(all_y, agg="sum", which_row=None):
    ys = []
    for y in all_y:
        y = np.asarray(y, dtype=float)
        if   agg == "sum":  ys.append(float(np.sum(y)))
        elif agg == "mean": ys.append(float(np.mean(y)))
        elif agg == "min":  ys.append(float(np.min(y)))
        elif agg == "max":  ys.append(float(np.max(y)))
        elif agg == "row":
            if which_row is None:
                raise ValueError("which_row must be provided when agg='row'.")
            ys.append(float(y[which_row]))
        else:
            raise ValueError("Unknown agg.")
    return np.array(ys, dtype=float)

def plot_two_panel_a_vs_y_v3(
    a, all_y, agg="sum", which_row=None,
    bins_total=40, savepath="fig_two_panel_a_vs_y_v3.png",
    show_corr=True
):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] < 1:
        raise ValueError("`a` must be (N, D).")
    a_last = a[:, -1]
    y_agg  = np.array(all_y).sum(-1)

    mask_pos = a_last > 0
    mask_neg = ~mask_pos
    n        = a_last.size
    cnt_neg  = int(mask_neg.sum())
    cnt_pos  = int(mask_pos.sum())
    cnt_zero = int(np.sum(a_last == 0))
    frac_neg, frac_zero, frac_pos = cnt_neg/n, cnt_zero/n, cnt_pos/n

    def _safe_corr(x, y):
        if x.size == 0 or y.size == 0 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
            return np.nan, np.nan
        r_p = np.corrcoef(x, y)[0, 1]
        r_s, _ = spearmanr(x, y)
        return r_p, r_s
    r_p, r_s = _safe_corr(a_last, y_agg)
    try:
        if cnt_neg>0 and cnt_pos>0:
            U, p_u = mannwhitneyu(y_agg[mask_neg], y_agg[mask_pos], alternative="two-sided")
        else:
            U, p_u = np.nan, np.nan
    except Exception:
        U, p_u = np.nan, np.nan
    med_neg = np.median(y_agg[mask_neg]) if cnt_neg else np.nan
    med_pos = np.median(y_agg[mask_pos]) if cnt_pos else np.nan

    # --- layout: use constrained_layout to leave room for text above axes
    fig, (ax_hist, ax_scatter) = plt.subplots(
        1, 1, figsize=(9.2, 4.8), constrained_layout=True
    )

    # ---- Left: histogram (neg/pos separate, zero as its own bar) ----
    nonzero = a_last[a_last != 0]
    if nonzero.size == 0:
        raise ValueError("All a_last are zero; cannot form histogram.")
    lo = min(nonzero.min(), 0.0)
    hi = max(nonzero.max(), 0.0)

    # histogram as light background
    ax_hist.hist(a_last[a_last < 0], bins=40, range=(lo, 0),
                alpha=0.4, color="#1f77b4", edgecolor="black", linewidth=0.5, label="a < 0")
    ax_hist.hist(a_last[a_last > 0], bins=40, range=(0, hi),
                alpha=0.4, color="#ff7f0e", edgecolor="black", linewidth=0.5, label="a ≥ 0")
    # zero mass as standalone bar
    ax_hist.bar(lo - (hi-lo)*0.05, cnt_zero, width=(hi-lo)*0.02,
                color="gray", edgecolor="black", linewidth=0.6, alpha=0.8, label="a = 0")

    # KDE smooth density curves
    x_grid = np.linspace(lo, hi, 500)
    if (a_last[a_last < 0]).size > 10:
        kde_neg = gaussian_kde(a_last[a_last < 0])
        ax_hist.plot(x_grid, kde_neg(x_grid)*len(a_last)*(hi-lo)/40, color="#1f77b4", lw=1.5)
    if (a_last[a_last > 0]).size > 10:
        kde_pos = gaussian_kde(a_last[a_last > 0])
        ax_hist.plot(x_grid, kde_pos(x_grid)*len(a_last)*(hi-lo)/40, color="#ff7f0e", lw=1.5)

    # vertical markers
    ax_hist.axvline(0.0, color="black", linestyle="--", lw=1.0)
    ax_hist.axvline(np.median(a_last), color="red", linestyle=":", lw=1.2)

    ax_hist.set_xlabel(r"$a_{\mathrm{last}}$")
    ax_hist.set_ylabel("Count / Density (scaled)")
    ax_hist.set_title(r"Distribution & Density of $a_{\mathrm{last}}$")
    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_hist.legend(frameon=False, loc="upper left")

    # move P-text above plot
    p_text = f"P(a<0)={frac_neg:.1%}   P(a=0)={frac_zero:.1%}   P(a>0)={frac_pos:.1%}"
    ax_hist.annotate(p_text, xy=(0.5, 1.06), xycoords='axes fraction',
                    ha='center', va='bottom', fontsize=10)

    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    return savepath


def plot_count_vs_ymean_pretty_v2(
    a, all_y, 
    agg="sum", 
    bins_total=40, 
    savepath="fig_count_vs_ymean_pretty_v2.png",
):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] < 1:
        raise ValueError("`a` must be (N, D).")
    a_last = a[:, -1]

    y = np.asarray(all_y)
    if y.ndim == 1:
        y_agg = y
    elif y.ndim >= 2:
        y_agg = y.sum(axis=-1) if agg == "sum" else y.mean(axis=-1)
    else:
        raise ValueError("`all_y` shape invalid.")

    # ---------- 分箱 ----------
    counts, bin_edges = np.histogram(a_last, bins=bins_total)
    bin_idxs = np.digitize(a_last, bin_edges, right=False)
    bin_idxs = np.minimum(bin_idxs, bins_total)

    y_mean = np.full(bins_total, np.nan)
    for b in range(1, bins_total + 1):
        mask = (bin_idxs == b)
        if np.any(mask):
            y_mean[b - 1] = np.mean(y_agg[mask])

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths  = np.diff(bin_edges)

    # ---------- 连续化 y_mean （插值） ----------
    if np.any(np.isnan(y_mean)):
        y_mean_interp = pd.Series(y_mean).interpolate(limit_direction="both").to_numpy()
    else:
        y_mean_interp = y_mean.copy()

    # ---------- 画图 ----------
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()

    # --- 样式: 去掉上框，但保留右轴 ---
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    # 保留右轴线但淡化
    ax2.spines["right"].set_linewidth(1.0)
    ax2.spines["right"].set_alpha(0.6)

    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    # --- 柱状图：区分 <0 和 ≥0，添加斜线填充 ---
    bin_mask_neg = bin_centers < 0
    bin_mask_pos = bin_centers >= 0
    color_neg = "#5B8FF9"  # 蓝
    color_pos = "#F4664A"  # 红

    bar_neg = ax1.bar(
        bin_centers[bin_mask_neg], counts[bin_mask_neg],
        width=bin_widths[bin_mask_neg], align="center",
        color=color_neg, alpha=0.5, edgecolor="black", linewidth=0.6,
        hatch="//", label="a < 0"
    )
    bar_pos = ax1.bar(
        bin_centers[bin_mask_pos], counts[bin_mask_pos],
        width=bin_widths[bin_mask_pos], align="center",
        color=color_pos, alpha=0.5, edgecolor="black", linewidth=0.6,
        hatch="\\\\", label="a ≥ 0"
    )

    ax1.set_ylabel("Count", fontsize=12)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(axis='both', labelsize=10)

    # --- 折线: 使用插值后的连续曲线 ---
    line_mean, = ax2.plot(
        bin_centers, y_mean_interp, color="#252525", linewidth=2.2, marker="o",
        markersize=4.5, label="Mean(y) per bin"
    )
    ax2.set_ylabel("Mean(y)", fontsize=12)
    ax2.tick_params(axis='y', labelsize=10)

    # --- 坐标轴线条美化 ---
    for spine in ["left", "bottom"]:
        ax1.spines[spine].set_linewidth(1.2)
    ax1.spines["left"].set_color("black")
    ax1.spines["bottom"].set_color("black")
    ax2.spines["right"].set_color("black")

    ax1.set_xlabel(r"$a_{\mathrm{last}}$ (binned)", fontsize=12)
    ax1.set_title(r"Distribution of $a_{2,3}$ and mean of $y$", fontsize=13, pad=10)

    # --- 图例 ---
    handles = [bar_neg, bar_pos, line_mean]
    labels = ["a < 0", "a ≥ 0", "Mean(y) per bin"]
    ax1.legend(handles, labels, frameon=False, loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)
    return savepath


def solve_params_from_sigmoid_eq(X, y, eps=1e-4):
    """
    解 a 于系统:
        y_k = sigmoid( sum_i X[k,i] * a_i ), k=0..5
    其中 sigmoid(t)=1/(1+exp(-t))

    参数:
        X: shape (6,6) 满秩矩阵
        y: shape (6,)    观测到的目标值, 要求 0<y<1

    返回:
        a: shape (6,), 解出的参数
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    y_clip = np.clip(y, eps, 1.0 - eps)

    # 检查维度
    assert X.shape == (6,6), "X必须是6x6矩阵"
    assert y.shape == (6,), "y必须是长度为6的一维向量"

    # 1. 计算 b_k = logit(y_k) = log(y_k/(1-y_k))
    b = np.log(y_clip / (1.0 - y_clip))  # 形状 (6,)

    # 2. 解线性方程组 X a = b
    #   X是满秩，所以可以直接用solve而不是最小二乘
    a = np.linalg.solve(X, b)  # 形状 (6,)
    return a

def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def papre_eval():
    mapping = {
        "0": json.load(open("/text2sql/verl-tool/test_outputs/whole/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6debug/searchR1_nq_results_100.json")),
        "1": json.load(open("/text2sql/verl-tool/test_outputs/whole/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyQuery/searchR1_nq_results_100.json")),
        "2": json.load(open("/text2sql/verl-tool/test_outputs/whole/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyAns/searchR1_nq_results_100.json")),
        "3": json.load(open("/text2sql/verl-tool/test_outputs/split/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyQueryQwen2.5-3B-Instruct/searchR1_nq_results_100.json")),
        "4": json.load(open("/text2sql/verl-tool/test_outputs/split/Qwen2.5-3Bsearch_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyAns/searchR1_nq_results_100.json")),
        "5": json.load(open("/text2sql/verl-tool/test_outputs/whole/Qwen2.5-3B/searchR1_nq_results_100.json"))
    }
    return mapping


def build_y(X, mapping, gt, idx):
    y = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        preds = mapping[str(i)][idx]['choices']
        scores = []
        for pred in preds:
            try:
               p = pred['message']['content']
            except:
               p = pred['text']
            s = compute_score(p, gt)
            scores.append(s)
        y[i] = sum(scores)/ len(scores)
    return y

data_source = 'searchR1_nq'
data = pd.read_parquet("/text2sql/verl-tool/data/searchR1_processed_direct/test.parquet")
selected_data = data[data.data_source == data_source]

X = np.array([
    [1,1,1,1,1,1], # both
    [1,1,0,1,0,0], # tool_use
    [1,0,1,0,1,0], # reasoning
    [1,1,0,0,0,0], # tool_use+raw
    [1,0,1,0,0,0], # raw+reasoning
    [1,0,0,0,0,0], # raw
], dtype=float)

mapping = papre_eval()

a, all_y = [],[]
for idx in tqdm(range(selected_data.shape[0])):
    y = build_y(X, mapping, data.reward_model[idx]["ground_truth"], idx)
    if y.sum() <= 0:
        continue
    a_hat = solve_params_from_sigmoid_eq(X, y)
    a.append(a_hat)
    all_y.append(y)

a = np.array(a)
# all_y = list_of_arrays # each element y has shape (6,)
# 1) histogram that clearly shows "mostly negative"
