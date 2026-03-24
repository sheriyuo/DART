#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析“跨能力(工具↔答案)”与“同能力↔同能力”的梯度余弦，给出层级统计、显著性检测和图表。
用法：
  python analyze_grad_cos.py --cross grad_cos_0.json --same grad_cos_sample.json --outdir figs_grad_cos
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, gaussian_kde
import seaborn as sns 


# ---------- 解析参数名 → (layer, comp) ----------
LAYER_RE = re.compile(r"layers\.(\d+)")

def parse_param(pname: str):
    """返回 (layer:int, comp:str)。未匹配到层则用 -1 表示 embeddings/head 等。"""
    m = LAYER_RE.search(pname)
    layer = int(m.group(1)) if m else -1
    if "embed_tokens" in pname: comp = "embed"
    elif "lm_head" in pname: comp = "lm_head"
    elif "self_attn.q_proj" in pname: comp = "attn_q"
    elif "self_attn.k_proj" in pname: comp = "attn_k"
    elif "self_attn.v_proj" in pname: comp = "attn_v"
    elif "self_attn.o_proj" in pname: comp = "attn_o"
    elif ".mlp.gate_proj" in pname: comp = "mlp_gate"
    elif ".mlp.up_proj" in pname: comp = "mlp_up"
    elif ".mlp.down_proj" in pname: comp = "mlp_down"
    else: comp = "other"
    return layer, comp


# ---------- 读取 JSON/JSONL ----------
_SKIP_KEYS = {"rid", "search_loss", "answer_loss", "loss_1", "loss_2", "acc"}

def _load_json_array(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data

def _load_jsonl(path: str):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def load_any_json(path: str):
    """自动识别 JSON 列表 或 JSONL"""
    with open(path, "r") as f:
        head = f.read(1)
    if head == "[":
        return _load_json_array(path)
    else:
        return _load_jsonl(path)

def to_dataframe(records):
    """把原始记录展开为 (rid, param, layer, comp, cos, angle_deg) 的 DataFrame"""
    rows = []
    for rec in records:
        rid = rec.get("rid")
        for k, v in rec.items():
            if k in _SKIP_KEYS:
                continue
            if isinstance(v, (int, float)):
                layer, comp = parse_param(k)
                cosv = float(v)
                # 数值保底
                if np.isnan(cosv):
                    continue
                cosv = float(np.clip(cosv, -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cosv)))
                rows.append({
                    "rid": rid,
                    "param": k,
                    "layer": layer,
                    "comp": comp,
                    "cos": cosv,
                    "angle_deg": ang,
                })
    return pd.DataFrame(rows)


# ---------- 统计/作图 ----------
def layer_summary(df):
    g = df.groupby("layer")["angle_deg"]
    return g.median(), g.quantile(0.25), g.quantile(0.75)

def plot_layer_curves(df_cross, df_same, outpath):
    med_c, q1_c, q3_c = layer_summary(df_cross)
    med_s, q1_s, q3_s = layer_summary(df_same)
    layers = sorted(set(med_c.index).union(med_s.index))
    x = layers
    mc = [med_c.get(l, np.nan) for l in x]
    ms = [med_s.get(l, np.nan) for l in x]
    c1 = [q1_c.get(l, np.nan) for l in x]; c3 = [q3_c.get(l, np.nan) for l in x]
    s1 = [q1_s.get(l, np.nan) for l in x]; s3 = [q3_s.get(l, np.nan) for l in x]

    plt.figure(figsize=(8,4))
    plt.plot(x, mc, marker="o", label="Cross (Tool↔Answer)")
    plt.fill_between(x, c1, c3, alpha=0.15)
    plt.plot(x, ms, marker="o", linestyle="--", label="Same-ability")
    plt.fill_between(x, s1, s3, alpha=0.15)
    plt.xlabel("Layer")
    plt.ylabel("Angle")
    plt.title("Layer-wise gradient angle: Cross vs Same")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_delta_with_sig(pair_df, outpath):
    layers = sorted(pair_df["layer"].unique())
    stats_rows = []
    for l in layers:
        sub = pair_df[pair_df["layer"] == l]
        if len(sub) >= 10:
            try:
                stat, p = wilcoxon(sub["angle_deg_cross"], sub["angle_deg_same"],
                                   alternative="greater")
            except ValueError:
                p = np.nan
        else:
            p = np.nan
        stats_rows.append({
            "layer": l,
            "delta_med": np.nanmedian(sub["angle_deg_cross"] - sub["angle_deg_same"]) if len(sub) > 0 else np.nan,
            "p": p, "n": len(sub)
        })
    stats = pd.DataFrame(stats_rows)

    plt.figure(figsize=(8,4))
    plt.bar(stats["layer"], stats["delta_med"])
    for _, r in stats.iterrows():
        p = r["p"]
        if isinstance(p, float) and not np.isnan(p):
            star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else ""))
            if star:
                plt.text(r["layer"], r["delta_med"], star, ha="center", va="bottom", fontsize=10)
    plt.axhline(0, linestyle=":")
    plt.xlabel("Layer")
    plt.ylabel("Δ angle (Cross − Same, degrees)")
    plt.title("Angle gap per layer (paired by (rid, param))")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_rho_curves(outpath):
    def rho(r, theta_deg):
        th = np.radians(theta_deg)
        num = 1 + r*np.cos(th)
        den = np.sqrt(1 + r*r + 2*r*np.cos(th))
        return num/den

    thetas = np.linspace(0, 90, 181)
    plt.figure(figsize=(6,4))
    for r in [0.5, 1.0, 2.0]:
        plt.plot(thetas, [rho(r,t) for t in thetas], label=f"r={r}")
    plt.xlabel("θ between g2 and g3 (degrees)")
    plt.ylabel("Progress ratio ρ(r,θ)")
    plt.title("Even with θ<90°, task-wise progress is reduced under step-norm limits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def _kde(y, x_grid, bw=None):
    if len(y) < 5:  # 太少不做KDE
        return np.zeros_like(x_grid)
    kde = gaussian_kde(y, bw_method=bw)  # bw=None 用Scott
    return kde(x_grid)

def plot_layer_ridge(df_cross, df_same, outpath,
                     angle_xlim=(0, 120),
                     min_n_per_layer=20,
                     y_step=1.2,
                     bw=None,
                     exclude_allzero=True):
    """
    分层Ridge图：每层两条密度曲线（Cross vs Same），堆叠展示。
    - exclude_allzero: 已在读入阶段剔除了“整条样本全0”，这里不再额外剔除。
    """
    # 角度网格
    x = np.linspace(angle_xlim[0], angle_xlim[1], 512)

    layers = sorted(set(df_cross["layer"]).intersection(set(df_same["layer"])))
    # 保留样本数充足的层
    kept_layers = []
    for l in layers:
        n1 = (df_cross["layer"]==l).sum()
        n2 = (df_same["layer"]==l).sum()
        if min(n1, n2) >= min_n_per_layer:
            kept_layers.append(l)
    if not kept_layers:
        raise RuntimeError("没有样本数达标的层，调小 min_n_per_layer 再试。")

    plt.figure(figsize=(8, 0.35*len(kept_layers)+1.8))
    base_y = 0.0

    for i, l in enumerate(kept_layers):
        a1 = df_cross.loc[df_cross["layer"]==l, "angle_deg"].values
        a2 = df_same.loc[df_same["layer"]==l,  "angle_deg"].values

        d1 = _kde(a1, x, bw=bw); 
        d2 = _kde(a2, x, bw=bw)
        # 为方便对比，归一化到相同峰高
        if d1.max() > 0: d1 = d1 / d1.max()
        if d2.max() > 0: d2 = d2 / d2.max()

        # 叠山：y 方向做平移
        plt.fill_between(x, base_y, base_y + d2, alpha=0.25)       # Same
        plt.plot(x, base_y + d2, linestyle="--", label=None)
        plt.fill_between(x, base_y, base_y + d1, alpha=0.25)       # Cross
        plt.plot(x, base_y + d1, label=None)

        # 层标签
        plt.text(angle_xlim[0]-5, base_y + 0.5, f"L{l}", va="center", ha="right", fontsize=9)
        base_y += y_step

    # 轴样式
    plt.xlim(angle_xlim)
    plt.yticks([]); plt.xlabel("Angle (degrees)")
    plt.title("Per-layer angle distributions (Ridge): Cross vs Same")
    # 绘制一组示例图例（用最后一层的曲线样式）
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0],[0], color="C0"), Line2D([0],[0], color="C0", linestyle="--")
    ]
    plt.legend(legend_lines, ["Cross (Tool↔Answer)", "Same-ability"], loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_aggregated_distribution(df_cross, df_same, outpath, 
                                 angle_xlim=(0, 120), 
                                 bw=None):
    """
    汇总全层数据：绘制 Cross vs Same 的全局分布对比图。
    """
    # 1. 设置样式
    sns.set_theme(style="whitegrid")
    # plt.rcParams['font.sans-serif'] = ['Arial'] # 确保字体清晰
    
    color_cross = "#2b7bba"  # 蓝色
    color_same = "#e37222"   # 橙色
    
    # 2. 提取全量数据
    data_cross = df_cross["angle_deg"].values
    data_same = df_same["angle_deg"].values
    
    # 计算全局均值
    mean_cross = np.mean(data_cross)
    mean_same = np.mean(data_same)
    
    # 3. 创建画布
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tick_params(axis='both', which='major', labelsize=12, 
                   colors='black', direction='out', length=6, width=1.5)
    ax.grid(False)
    
    # 4. 绘制密度图 (KDE)
    # 使用 seaborn 的 kdeplot 更简便且美化效果好
    sns.kdeplot(data_cross, fill=True, color=color_cross, label="Cross (Tool↔Answer)", 
                alpha=0.4, linewidth=2, bw_adjust=1 if bw is None else bw)
    sns.kdeplot(data_same, fill=True, color=color_same, label="Same-ability", 
                alpha=0.3, linewidth=2, bw_adjust=1 if bw is None else bw)
    
    # # 5. 标注均值线
    # # 绘制垂直线
    # ax.axvline(mean_cross, color=color_cross, linestyle='-', linewidth=2, alpha=0.8)
    # ax.axvline(mean_same, color=color_same, linestyle='--', linewidth=2, alpha=0.8)
    
    # # 添加均值文本标注
    # # 动态计算文本高度（放在 y 轴最大值的 90% 处）
    # y_max = ax.get_ylim()[1]
    # ax.text(mean_cross + 2, y_max * 0.9, f"Mean: {mean_cross:.2f}°", 
    #         color=color_cross, fontweight='bold', fontsize=10)
    # ax.text(mean_same - 2, y_max * 0.8, f"Mean: {mean_same:.2f}°", 
    #         color=color_same, fontweight='bold', fontsize=10, ha='right')

    # 6. 轴体美化
    ax.set_xlim(angle_xlim)
    # ax.set_xlabel("Angle (degrees)", fontsize=12)
    # ax.set_ylabel("Density", fontsize=12)
    # ax.set_title("Aggregated Angle Distribution: Cross vs Same", fontsize=14, pad=20, fontweight='bold')
    
    # 移除顶部和右侧边框
    sns.despine()
    
    # 添加图例
    # ax.legend(frameon=False, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    print(f"Aggregated plot saved to {outpath}")
    plt.close()


def plot_layer_facets(
    df_cross, df_same, outpath,
    angle_xlim=(0, 120),
    min_n_per_layer=20,
    stride=1,
    bw=None,
    max_layers_to_plot=5,
):
    # 1. 选出层（逻辑与原代码一致）
    all_layers = sorted(set(df_cross["layer"]).intersection(set(df_same["layer"])))
    layers = [l for l in all_layers if l % stride == 0]
    layers = [
        l for l in layers
        if (df_cross["layer"] == l).sum() >= min_n_per_layer
        and (df_same["layer"] == l).sum() >= min_n_per_layer
    ]
    if not layers:
        raise RuntimeError("没有样本数达标的层，请降低 stride 或 min_n_per_layer。")

    # 2. 采样最多 max_layers_to_plot 个层
    if len(layers) > max_layers_to_plot:
        idx = np.linspace(0, len(layers) - 1, num=max_layers_to_plot, dtype=int)
        layers = [layers[i] for i in sorted(set(idx))]

    # 3. 准备角度网格和画布
    x = np.linspace(angle_xlim[0], angle_xlim[1], 512)
    fig, ax = plt.subplots(figsize=(8, 6))  # 增大图的尺寸，提升可读性

    # 改用seaborn的调色板，颜色更柔和且区分度高
    colors = sns.color_palette("husl", n_colors=len(layers))  
    linestyles = ["-", "--"]  # 实线代表Cross，虚线代表Same

    for i, l in enumerate(layers):
        a_cross = df_cross.loc[df_cross["layer"] == l, "angle_deg"].values
        a_same  = df_same.loc[df_same["layer"] == l, "angle_deg"].values

        # KDE + 归一化
        d_cross = _kde(a_cross, x, bw=bw)
        d_same  = _kde(a_same,  x, bw=bw)
        if d_cross.max() > 0:
            d_cross = d_cross / d_cross.max()
        if d_same.max() > 0:
            d_same = d_same / d_same.max()

        # 颜色区分层，线型区分任务类型
        ax.plot(
            x, d_cross,
            color=colors[i],
            linestyle=linestyles[0],
            label=f"L{l} Cross",
            linewidth=2  # 增加线宽，让曲线更醒目
        )
        ax.plot(
            x, d_same,
            color=colors[i],
            linestyle=linestyles[1],
            label=f"L{l} Same",
            linewidth=2
        )

    # 4. 轴范围与标注优化
    ax.set_xlim(angle_xlim)
    ax.set_ylim(bottom=0, top=1.1)  # 预留一点顶部空间，避免曲线顶到边界
    ax.set_xlabel("Angle (degrees)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Density", fontsize=12, fontweight="bold")
    ax.set_title("Angle Distributions Across Layers", fontsize=14, fontweight="bold", pad=20)

    # 去掉右边和上边框（保持原风格）
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # 添加网格线（辅助读数，增强可读性）
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # 图例优化：按层分组，调整位置和样式
    ax.legend(
        loc="upper right",
        fontsize=10,
        ncol=2,
        frameon=False,
        title="Layer / Task Type",  # 图例标题，明确分组逻辑
        title_fontsize=11
    )

    # 调整刻度标签大小
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")  # bbox_inches避免图例被截断
    plt.close(fig)


def remove_zero_angle_records(df, eps=1.0):
    """删除整条样本(rid)在所有层角度都为0的异常数据。"""
    # 每个样本的最大角度
    max_angle = df.groupby("rid")["angle_deg"].max().reset_index()
    valid_rids = max_angle[max_angle["angle_deg"] > eps]["rid"]
    before, after = df["rid"].nunique(), valid_rids.nunique()
    print(f"[Filter] 移除 {before - after} 个全层角度≈0°的异常样本，共保留 {after} 个。")
    return df[df["rid"].isin(valid_rids)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", default="7b_hotpot_gradres/grad_cos_group_search.json", help="跨能力文件路径（如 grad_cos_0.json/jsonl）")
    ap.add_argument("--same",  default="7b_hotpot_gradres/grad_cos_sample_search.json", help="同能力文件路径（如 grad_cos_sample.json/jsonl）")
    ap.add_argument("--outdir", default="7b_hotpot_gradres", help="图表输出目录")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 读取
    cross_raw = load_any_json(args.cross)
    same_raw  = load_any_json(args.same)

    df_cross = to_dataframe(cross_raw)
    df_same  = to_dataframe(same_raw)

    df_cross = remove_zero_angle_records(df_cross, eps=1.0)
    df_cross['rid'] = range(len(df_cross))
    df_same  = remove_zero_angle_records(df_same, eps=1.0)
    df_same['rid'] = range(len(df_same))

    if len(df_cross)==0 or len(df_same)==0:
        raise RuntimeError("读到的有效参数项为空，请检查文件路径与内容。")

    # 严格在 (rid, param, layer) 上配对
    pair = (df_cross[["rid","param","layer","angle_deg"]]
            .merge(df_same[["rid","param","layer","angle_deg"]],
                   on=["rid","param","layer"], suffixes=("_cross","_same")))
    if len(pair)==0:
        raise RuntimeError("两个文件在 (rid,param,layer) 上无交集，请确认实验 ID 与参数名一致。")

    # 图A：层级中位数曲线（含四分位带）
    plot_aggregated_distribution(
        df_cross, df_same,
        outpath=os.path.join(args.outdir, "fig_aggregated_distribution.png"),
        angle_xlim=(0, 120),
        bw=None
    )
#     plot_layer_curves(df_cross, df_same, os.path.join(args.outdir, "figA_layer_curves.png"))

#     # 图B：层级 Δangle + Wilcoxon 标星
#     plot_delta_with_sig(pair, os.path.join(args.outdir, "figB_layer_delta_sig.png"))

#     # 图C：ρ(r,θ) 理论曲线（说明“θ<90°也抑制”）
#     plot_rho_curves(os.path.join(args.outdir, "figC_rho_curves.png"))

#     plot_layer_ridge(
#     df_cross, df_same,
#     outpath=os.path.join(args.outdir, "fig_layer_ridge.png"),
#     angle_xlim=(0, 120),      # 视你的数据而定
#     min_n_per_layer=20,       # 每层至少多少样本才画
#     y_step=1.0,               # 层与层的垂直间距
#     bw=None                   # KDE 带宽，None=Scott；也可设 0.3/0.5 等相对平滑
# )

#     # Facets 分面（隔层采样，减少面板数）
#     plot_layer_facets(
#         df_cross, df_same,
#         outpath=os.path.join(args.outdir, "fig_layer_facets_stride2.png"),
#         angle_xlim=(0, 120),
#         min_n_per_layer=20,
#         stride=2,                 # 每两层画一个；设为1画所有层
#         bw=None
#     )

#     # 终端概要
#     overall_delta_med = np.median(pair["angle_deg_cross"] - pair["angle_deg_same"])
#     print(f"[Summary] 全局(配对)中位角度差 Δmedian = {overall_delta_med:.3f}°  (Cross − Same)")

#     # 前几层差值最大（便于论文填数字）
#     layer_stats = (pair.assign(d_angle=lambda d: d["angle_deg_cross"]-d["angle_deg_same"])
#                         .groupby("layer")["d_angle"].agg(["median","count"])
#                         .sort_values("median", ascending=False))
#     print("[Top layers by Δmedian (degrees)]:")
#     print(layer_stats.head(8).to_string())

    # 保存配对明细（可做附录或复核）
    pair.to_csv(os.path.join(args.outdir, "paired_angles_detail.csv"), index=False)
    print(f"[OK] 图表与明细已输出到: {args.outdir}")


if __name__ == "__main__":
    main()
