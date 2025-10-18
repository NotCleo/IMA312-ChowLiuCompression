#!/usr/bin/env python3
import os
import io
import gzip
import math
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

IP_POOL = [".".join(str(random.randint(0, 255)) for _ in range(4)) for _ in range(200)]
URL_POOL = ["/home", "/api/v1/users", "/api/v1/products", "/login", "/signup", "/profile", "/search?q=query", "/product/1234", "/cart", "/checkout", "/blog/post1", "/about", "/contact", "/faq", "/terms", "/privacy", "/api/v2/auth", "/dashboard", "/settings", "/logout", "/forgot-password", "/reset-password", "/category/electronics", "/category/books", "/category/clothing", "/item/5678", "/reviews", "/support", "/help", "/news", "/events", "/forum/thread/91011", "/wiki/page", "/download/file.pdf", "/upload", "/stream/video.mp4", "/image/gallery", "/video/watch", "/audio/listen", "/map/location", "/weather", "/stock/quote", "/currency/convert", "/calculator", "/translator", "/calendar", "/notes", "/tasks", "/reminders", "/alarms"]
USER_POOL = [f"user_{i}" for i in range(100)]
METHOD_POOL = ['GET', 'POST', 'PUT', 'DELETE']
PROTOCOL_POOL = ['HTTP/1.1', 'HTTP/2.0']
REFERER_POOL = ["-", "https://www.google.com/", "https://www.bing.com/", "https://example.com/", "https://facebook.com/"]
USER_AGENT_POOL = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36", "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0", "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.864.48 Safari/537.36 Edg/91.0.864.48", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36", "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"]

def generate_synthetic_logs(n_rows=5000, corr_strength=0.5, filename="log.csv"):
    np.random.seed(42)
    random.seed(42)
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(seconds=random.randint(1, 3600)) for _ in range(n_rows)]
    hours = np.array([t.hour for t in timestamps])
    ips = np.random.choice(IP_POOL, n_rows)
    users = np.random.choice(USER_POOL, n_rows)
    methods = np.random.choice(METHOD_POOL, n_rows, p=[0.7, 0.2, 0.05, 0.05])
    urls = np.random.choice(URL_POOL, n_rows)
    protocols = np.random.choice(PROTOCOL_POOL, n_rows, p=[0.8, 0.2])
    referers = np.random.choice(REFERER_POOL, n_rows, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    user_agents = np.random.choice(USER_AGENT_POOL, n_rows)
    statuses = []
    bytes_sent = []
    latency_ms = []
    for i in range(n_rows):
        m = methods[i]
        h = hours[i]
        error_prob = 0.05 + corr_strength * (0.3 if m in ['POST', 'PUT'] else 0.1) + (0.15 if 0 <= h < 6 else 0)
        if np.random.rand() < error_prob:
            status = np.random.choice([404, 500, 403], p=[0.5, 0.3, 0.2])
        else:
            status = 200
        mean_bytes = 5000 if status == 200 else 500
        mean_bytes += corr_strength * (3000 if m == 'POST' else -1500 if m == 'DELETE' else 500)
        bytes_val = max(0, int(np.random.normal(mean_bytes, 1000 / (corr_strength + 0.1))))
        mean_latency = 50 + corr_strength * (bytes_val / 50) + corr_strength * (100 if 12 <= h < 18 else 0)
        lat_val = max(10, int(np.random.normal(mean_latency, 30 / (corr_strength + 0.1))))
        statuses.append(status)
        bytes_sent.append(bytes_val)
        latency_ms.append(lat_val)
    df = pd.DataFrame({
        "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "ip": ips,
        "user": users,
        "method": methods,
        "url": urls,
        "protocol": protocols,
        "status": statuses,
        "bytes_sent": bytes_sent,
        "latency_ms": latency_ms,
        "referer": referers,
        "user_agent": user_agents
    })
    df.to_csv(filename, index=False)
    print(f"\033[92m[+] Generated {filename}\033[0m  ({n_rows} rows, corr_strength={corr_strength:.2f})")
    return df

def compute_empirical_probabilities(df, col1, col2=None):
    if col2 is None:
        counts = df[col1].value_counts()
        probs = counts / counts.sum()
        return probs.to_dict()
    else:
        joint = df.groupby([col1, col2]).size()
        joint = joint / joint.sum()
        return joint.to_dict()

def encoding_cost_bits(df, col1, col2):
    pxy = compute_empirical_probabilities(df, col1, col2)
    pickled = pickle.dumps(pxy)
    compressed = gzip.compress(pickled, compresslevel=9)
    bytes_size = len(compressed)
    return bytes_size * 8

def mutual_information(df, col1, col2):
    if col1 == col2:
        return 0.0
    pxy = compute_empirical_probabilities(df, col1, col2)
    px = compute_empirical_probabilities(df, col1)
    py = compute_empirical_probabilities(df, col2)
    mi = 0.0
    for (x, y), p_xy in pxy.items():
        if p_xy > 0:
            mi += p_xy * math.log(p_xy / (px.get(x, 1e-12) * py.get(y, 1e-12) + 1e-12) + 1e-12, 2)
    return max(mi, 0.0)

def learn_chow_liu_tree(df):
    cols = df.columns.tolist()
    G = nx.Graph()
    n = len(df)
    print("   ↳ Computing pairwise penalized mutual information...")
    n_cols = len(cols)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col1, col2 = cols[i], cols[j]
            mi = mutual_information(df, col1, col2)
            cost_bits = encoding_cost_bits(df, col1, col2)
            penalized_mi = mi - (cost_bits / n)
            G.add_edge(col1, col2, weight=penalized_mi)
    T = nx.maximum_spanning_tree(G, weight='weight', algorithm='kruskal')
    return G, T

def empirical_entropy(prob_dict):
    return -sum(p * math.log(p + 1e-12, 2) for p in prob_dict.values() if p > 0)

def estimate_chow_liu_compression(df, T):
    cols = df.columns.tolist()
    root = cols[0]
    directed_T = nx.bfs_tree(T, root)
    total_MI = 0.0
    for u, v in T.edges():
        total_MI += mutual_information(df, u, v)
    total_entropy = sum(empirical_entropy(compute_empirical_probabilities(df, c)) for c in cols)
    n = len(df)
    data_bits = n * (total_entropy - total_MI)
    data_bytes = data_bits / 8.0
    model = {'root': root, 'edges': list(directed_T.edges()), 'probs': {}}
    model['probs'][root] = compute_empirical_probabilities(df, root)
    for parent, child in directed_T.edges():
        cond_probs = {}
        parent_vals = df[parent].unique()
        for pv in parent_vals:
            subset = df[df[parent] == pv]
            if len(subset) == 0:
                continue
            cond = compute_empirical_probabilities(subset, child)
            cond_probs[pv] = cond
        model['probs'][(parent, child)] = cond_probs
    pickled_model = pickle.dumps(model)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(pickled_model)
    model_bytes = len(buf.getvalue())
    total_bytes = data_bytes + model_bytes
    return total_bytes, data_bytes, model_bytes

def gzip_compress_size(file_path):
    orig_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        compressed_data = gzip.compress(f_in.read(), compresslevel=9)
    comp_size = len(compressed_data)
    return orig_size, comp_size

def visualize_tree(T, filename):
    pos = nx.spring_layout(T, seed=42, k=0.5, iterations=50)
    fig, ax = plt.subplots(figsize=(10, 7))
    weights = [T[u][v]['weight'] for u, v in T.edges()]
    nx.draw(T, pos, with_labels=True, node_color='lightblue', node_size=2000, edge_color=weights, width=4.0, edge_cmap=plt.cm.Blues, font_size=10, font_weight='bold', ax=ax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(weights or [0]), vmax=max(weights or [1])))
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Penalized Mutual Information (bits)", rotation=270, labelpad=15)
    ax.set_title("Penalized Chow–Liu Dependency Tree", fontsize=14)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ↳ Tree visualization saved as: {filename}")

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} P{suffix}"

def main():
    os.makedirs("logs", exist_ok=True)
    os.chdir("logs")
    datasets = [("log_low.csv", 0.05, 1500), ("log_medium.csv", 0.60, 12000), ("log_high.csv", 0.95, 45000)]
    results = []
    for fname, corr, n_rows in datasets:
        print(f"\n\033[94m[Dataset] {fname} (corr={corr}, rows={n_rows})\033[0m")
        df = generate_synthetic_logs(n_rows, corr, fname)
        G, T = learn_chow_liu_tree(df)
        visualize_tree(T, fname.replace(".csv", "_tree.png"))
        total_chow, data_chow, model_chow = estimate_chow_liu_compression(df, T)
        orig_size, gzip_size = gzip_compress_size(fname)
        results.append({"Dataset": fname, "Rows": n_rows, "Corr": corr, "Original": orig_size, "GZIP": gzip_size, "ChowLiu_Total": total_chow, "ChowLiu_Data": data_chow, "ChowLiu_Model": model_chow})
    res_df = pd.DataFrame(results)
    print("\n\033[95m===== Compression Summary =====\033[0m")
    print(res_df[['Dataset', 'Rows', 'Corr', 'Original', 'GZIP', 'ChowLiu_Total']].to_string(index=False))
    print("\nHuman-readable sizes:")
    for _, row in res_df.iterrows():
        print(f"{row['Dataset']:15} | Rows={int(row['Rows']):5d} | Corr={row['Corr']:.2f} | Orig={sizeof_fmt(row['Original'])} | GZIP={sizeof_fmt(row['GZIP'])} | Chow–Liu total={sizeof_fmt(row['ChowLiu_Total'])} (data={sizeof_fmt(row['ChowLiu_Data'])}, model={sizeof_fmt(row['ChowLiu_Model'])})")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(res_df))
    width = 0.25
    plt.bar(x - width, res_df["Original"] / 1024, width, label="Original")
    plt.bar(x, res_df["GZIP"] / 1024, width, label="GZIP")
    plt.bar(x + width, res_df["ChowLiu_Total"] / 1024, width, label="Chow–Liu (est. total)")
    plt.xticks(x, res_df["Dataset"], rotation=20)
    plt.ylabel("File Size (KB)")
    plt.title("Compression Comparison: Original vs GZIP vs Penalized Chow–Liu")
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("compression_summary.png", dpi=150)
    plt.close()
    print("\033[92m[+] Saved comparison chart as compression_summary.png\033[0m")
    res_df['GZIP_Ratio'] = res_df['GZIP'] / res_df['Original'] * 100
    res_df['ChowLiu_Ratio'] = res_df['ChowLiu_Total'] / res_df['Original'] * 100
    print("\n\033[95m===== Compression Ratios (%) =====\033[0m")
    print(res_df[['Dataset', 'GZIP_Ratio', 'ChowLiu_Ratio']].to_string(index=False, float_format="%.2f"))

if __name__ == "__main__":
    main()
