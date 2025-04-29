import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

#  Paul Tol's Colorblind-Friendly Palette
TASK_COLORS = ['#332288', '#88CCEE', '#44AA99', '#DDCC77']  # 4-color safe palette

def preprocess_for_plotting(df):
    """
    Normalize the dataframe for plotting (only min-max scaling, no inversion).
    """
    df_processed = df.copy()

    # Perform min-max scaling for selected numeric columns
    for col in ['avg_score', 'avg_latency', 'avg_memory_kb', 'tokens_per_second']:
        min_val = df_processed[col].min()
        max_val = df_processed[col].max()
        df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val + 1e-8)

    return df_processed

def set_plotting_style():
    sns.set(style="whitegrid")
    sns.set_palette(TASK_COLORS)
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })


def create_visualizations(summary, results_dir='results/plots'):
    os.makedirs(results_dir, exist_ok=True)
    set_plotting_style()

    df = flatten_summary(summary)
    df_normalized = preprocess_for_plotting(df)  # Min-max scaled version for radar and scatter plots

    # ➡ Use raw df for bar charts, dashboard, and heatmaps
    create_bar_chart(df, 'avg_score', 'Performance Comparison', results_dir)
    create_bar_chart(df, 'avg_latency', 'Latency Comparison', results_dir)
    create_bar_chart(df, 'avg_memory_kb', 'Memory Usage Comparison', results_dir)
    create_bar_chart(df, 'tokens_per_second', 'Tokens per Second Comparison', results_dir)

    create_performance_dashboard(df, results_dir)
    create_enhanced_heatmap(summary, results_dir)

    # ➡ Use normalized df for radar charts and scatter plots
    create_performance_vs_speed_scatter(df_normalized, results_dir)
    create_enhanced_radar_chart(df_normalized, results_dir)



def flatten_summary(summary):
    rows = []
    for model, tasks in summary.items():
        for task, metrics in tasks.items():
            row = {'Model': model, 'Task': task}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def create_bar_chart(df, metric, title, results_dir):
    plt.figure(figsize=(10, 6))

    # Remove rows where the metric is zero or very close to zero
    plot_df = df.copy()
    plot_df = plot_df[plot_df[metric] > 1e-6]

    sns.barplot(data=plot_df, x="Model", y=metric, hue="Task", palette=TASK_COLORS, errorbar=None)
    plt.title(title)
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("Model")
    plt.legend(title="Task")

    for bar in plt.gca().patches:
        height = bar.get_height()
        if height >= 0.01:  # Only annotate bars with significant height
            plt.gca().annotate(f"{height:.2f}", (bar.get_x() + bar.get_width()/2., height),
                               ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()



def create_performance_dashboard(df, results_dir):
    """
    Create a performance dashboard showing Score, Latency, Memory Usage, and Tokens per Second.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # 2x2 grid layout

    metrics = ['avg_score', 'avg_latency', 'avg_memory_kb', 'tokens_per_second']
    titles = ['Score', 'Latency (s)', 'Memory Usage (KB)', 'Tokens per Second']

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        plot_df = df.copy()
        plot_df = plot_df[plot_df[metric] > 1e-6]  #  Remove near-zero entries

        sns.barplot(data=plot_df, x='Model', y=metric, hue='Task', ax=ax, palette=TASK_COLORS, errorbar=None)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('')

        for bar in ax.patches:
            height = bar.get_height()
            if height >= 0.01:
                ax.annotate(f"{height:.2f}", (bar.get_x() + bar.get_width()/2., height),
                            ha='center', va='bottom', fontsize=10)

    plt.suptitle("Performance Dashboard", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "performance_dashboard.png"))
    plt.close()




def create_performance_vs_speed_scatter(df, results_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="avg_latency", y="avg_score", hue="Model", style="Task", s=100, palette=TASK_COLORS)
    plt.title("Performance vs Latency")
    plt.xlabel("Latency (normalized)")
    plt.ylabel("Score (normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "performance_vs_latency_scatter.png"))
    plt.close()


def create_enhanced_radar_chart(df, results_dir):
    metrics = ['avg_score', 'avg_latency', 'avg_memory_kb']
    models = df['Model'].unique()
    task_order = ['qa', 'code', 'summarization', 'reasoning']

    for model in models:
        radar_data = df[df['Model'] == model].set_index('Task')[metrics]
        radar_data = radar_data.reindex(task_order).fillna(0.0)  # Fill missing tasks with 0

        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        for i, task in enumerate(radar_data.index):
            values = radar_data.loc[task].tolist()
            values += values[:1]
            ax.plot(angles, values, label=task, color=TASK_COLORS[i % len(TASK_COLORS)])
            ax.fill(angles, values, alpha=0.1, color=TASK_COLORS[i % len(TASK_COLORS)])

        plt.xticks(angles[:-1], metrics)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
        plt.title(f"Radar Chart - {model}", size=16, y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()

        safe_model_name = model.replace(":", "_").replace("/", "_")
        plt.savefig(os.path.join(results_dir, f"radar_chart_{safe_model_name}.png"))
        plt.close()



def create_enhanced_heatmap(summary, results_dir):
    rows = []
    for model, task_dict in summary.items():
        for task, metrics in task_dict.items():
            row = {'Model': model, 'Task': task, **metrics}
            rows.append(row)
    df = pd.DataFrame(rows)

    pivot_score = df.pivot(index="Model", columns="Task", values="avg_score")
    pivot_latency = df.pivot(index="Model", columns="Task", values="avg_latency")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_score, annot=True, cmap="cividis", fmt=".2f")
    plt.title("Score Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "score_heatmap.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_latency, annot=True, cmap="cividis", fmt=".2f")
    plt.title("Latency Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "latency_heatmap.png"))
    plt.close()
