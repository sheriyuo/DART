import json
import matplotlib.pyplot as plt

# 输入数据：替换成你的真实列表
data = json.load(open("search_r1_qa_em_lora1_tag_erro_stats.json", "r"))

steps = [d["step"] for d in data]
ratio_total_error = [d["total_erro_nums"] / d["total_records"] for d in data]
ratio_positive_error = [
    d["positive_erro_nums"] / d["total_erro_nums"] if d["total_erro_nums"] > 0 else 0
    for d in data
]

plt.figure(figsize=(8, 5))
plt.plot(steps, ratio_total_error, marker='o', label="total_erro_nums / 8192")
plt.plot(steps, ratio_positive_error, marker='o', label="positive_erro_nums / total_erro_nums")

plt.xlabel("step")
plt.ylabel("ratio")
plt.title("Error Ratios Over Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("error_ratio_plot.png", dpi=300)
