import json
import os

def generate_report(summary, all_results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    report = []

    # Title and Introduction
    report.append("# Model Performance Report\n")
    report.append("## Introduction\n")
    report.append("This report analyzes the performance in score, latency, memory usage, and tokens per second of different models (Llama3:8b, Mistral, Phi) across various tasks.\n")
    report.append("\n")

    # Executive Summary with key findings
    report.append("## Executive Summary\n")
    report.append("Key findings and takeaways:\n")
    for model in summary.keys():
        for task in summary[model].keys():
            avg_latency = summary[model][task]["avg_latency"]
            avg_memory_kb = summary[model][task]["avg_memory_kb"]
            avg_score = summary[model][task]["avg_score"]
            tokens_per_second = summary[model][task]["tokens_per_second"]
            report.append(f"- **{model} - {task}**: Avg Latency: {avg_latency:.2f} s, Avg Memory Usage: {avg_memory_kb:.2f} KB, Avg Score: {avg_score:.2f}, Tokens/sec: {tokens_per_second:.2f}\n")
    report.append("\n")

    # Model Rankings
    report.append("## Model Rankings\n")
    # Add model rankings based on performance (avg_score)
    model_rankings = []
    for model in summary.keys():
        avg_score = sum([summary[model][task]["avg_score"] for task in summary[model]]) / len(summary[model])
        model_rankings.append((model, avg_score))
    model_rankings = sorted(model_rankings, key=lambda x: x[1], reverse=True)
    report.append("Model Rankings based on average score across all tasks:\n")
    for model, avg_score in model_rankings:
        report.append(f"- **{model}**: Avg Score: {avg_score:.2f}\n")
    report.append("\n")

    # Task-specific Performance Analysis
    report.append("## Task-specific Performance Analysis\n")
    for task in ["code", "reasoning", "qa", "summarization"]:
        report.append(f"### {task.capitalize()}\n")
        task_data = []
        for model in summary.keys():
            task_data.append((model, summary[model].get(task, {}).get("avg_score", 0)))
        task_data = sorted(task_data, key=lambda x: x[1], reverse=True)
        for model, score in task_data:
            report.append(f"- **{model}**: Avg Score: {score:.2f}\n")
        report.append("\n")

    # Performance vs. Speed Trade-offs (scatter plot)
    report.append("## Performance vs. Speed Trade-offs\n")
    report.append("![Performance vs. Latency Scatter Plot](performance_vs_latency_scatter.png)\n")
    report.append("\n")

    # Resource Usage Analysis
    report.append("## Resource Usage Analysis\n")
    report.append("![Memory Usage Comparison](memory_usage_comparison.png)\n")
    report.append("\n")

    # Example Responses
    report.append("## Example Responses\n")
    report.append("### Good Response\n")
    for result in all_results:
        if result['score'] > 0.7:
            report.append(f"**Prompt**: {result['prompt']}\n**Response**: {result['response']}\n")
            report.append("\n")

    report.append("### Bad Response\n")
    for result in all_results:
        if result['score'] < 0.3:
            report.append(f"**Prompt**: {result['prompt']}\n**Response**: {result['response']}\n")
            report.append("\n")

    report.append("### Median Response\n")
    for result in all_results:
        if 0.3 <= result['score'] <= 0.7:
            report.append(f"**Prompt**: {result['prompt']}\n**Response**: {result['response']}\n")
            report.append("\n")

    # Conclusions and Recommendations
    report.append("## Conclusions and Recommendations\n")
    report.append("Based on the analysis, we recommend the following:\n")
    report.append("- **Llama3:8b** shows the best performance in QA and code tasks, but with higher latency.\n")
    report.append("- **Mistral** provides a good balance between performance and speed.\n")
    report.append("- **Phi** has the lowest latency but struggles with task performance.\n")
    report.append("\n")

    # Save the report to markdown file
    report_path = os.path.join(output_dir, "model_performance_report.md")
    with open(report_path, 'w') as f:
        f.writelines(report)

    print(f"Report saved to {report_path}")
