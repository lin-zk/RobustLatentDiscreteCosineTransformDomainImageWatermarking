from scipy.stats import binom
import math

def calculate_threshold(n, alpha=0.01):
    p = 0.5  # 无水印情况下的正确概率
    for k in range(n + 1):
        # 计算 P(X >= k)
        p_value = 1 - binom.cdf(k - 1, n, p)
        if p_value < alpha:
            return math.ceil(k)
    return n  # 如果没有找到阈值，返回最大值

n = 64  # 位数
alpha = 1e-2 # fpr
threshold = calculate_threshold(n, alpha)
print(f"For {n} bits, at least {threshold} correct bits are required to significantly prove the presence of a watermark (FPR={alpha}), which corresponds to a bit accuracy of {threshold / n:.4%}.")