import os
import json
from pathlib import Path

def generate_report(summary, output_dir='results'):
    # Creat results directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Start building the markdown content
    report = []

    # Title and Introduction
    report.append("# LLM Benchmarking Results\n")
    report.append("This report analyzes the performance, latency, and memory usage of different models (Llama3:8b, Mistral, Phi) across various tasks.\n")
    report.append("\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append("In terms of overall performance, all three models are not widely differentiated. Users must choose a suitable one for each specific task and practical requirement. Mistral performs the best in QA and reasoning, while Llama3:8b specializes in coding and summarization. Both models offer higher task accuracy at the expense of latency. For contexts that emphasize efficiency, Phi is recommended due to its low latency and high TPS.\n")
    report.append("\n")

    # Model Rankings
    report.append("## Model Rankings\n")
    report.append("![Performance Dashboard](../benchmark_results/performance_dashboard.png)\n")
    report.append("Llama3:8b and Mistral perform higher task scores, but to obtain a faster response, we recommend Phi.\n")
    report.append("\n")

    # Task-specific Performance Analysis
    report.append("## Task-specific Performance Analysis\n")
    report.append("![Score Heatmap](../benchmark_results/avg_score_heatmap.png)\n")
    report.append("![Latency Heatmap](../benchmark_results/avg_latency_heatmap.png)\n")
    report.append("![Memory Usage Heatmap](../benchmark_results/avg_memory_kb_heatmap.png)\n")
    report.append("All models perform well in QA while Mistral is specialized in reasoning considering both performance and speed. For coding and summarization, their performances are similar, but Phi specializes in speed.\n")

    # Performance vs. Speed Trade-offs (scatter plot)
    report.append("## Performance vs. Speed Trade-offs\n")
    report.append("![Performance vs. Latency Scatter Plot](../benchmark_results/performance_vs_latency_scatter.png)\n")
    report.append("From the scatter plot, there is a roughly reciprocal relationship between score and latency time, meaning that high score is correlated with low latency. However, apart from the three QA results, all the other points reside on the lower part of the graph, with no apparent trade-off. Nevertheless, we can still conclude that Phi delivers acceptable scores with low latency from the discovery that it dominates the left area.\n")
    report.append("\n")

    # Resource Usage Analysis
    report.append("## Resource Usage Analysis\n")
    report.append("![Memory Usage Comparison](../benchmark_results/memory_usage_comparison.png)\n")
    report.append("All these three models are optimized so that their memory usage across tasks is around 0.05KB. On average, Mistral makes the lowest memory consumption. There is an outlier that Phi's average memory usage excels at nearly 0.07KB for QA tasks, probably due to Phi's high speed in generation.\n")
    report.append("\n")

    # Example Responses
    report.append("## Example Responses\n")

    with open('results/llama3_8b_qa.json') as file:
        data = json.load(file)
    report.append("### Good Response\n")
    for result in data:
        if result['score'] >= 1:
            report.append('''
                          **Prompt**: {result['prompt']}\n
                          **Response**: {result['response']}\n
                          ''')
            report.append("\n")
            break

    with open('results/llama3_8b_reasoning.json') as file:
        data = json.load(file)
    report.append("### Bad Response\n")
    for result in data:
        if result['score'] < 0.3:
            report.append('''
                          **Prompt**: {result['prompt']}\n
                          **Response**: {result['response']}\n
                          ''')
            report.append("\n")
            break

    # Conclusions and Recommendations
    report.append("## Conclusions and Recommendations\n")
    report.append("No single model can universally outperform others in all tasks. Instead, the choice of the model should be based on task-specific requirements and performance priorities, such as memory and speed.\n")
    report.append("\n")

    # Save the report to markdown file
    report_path = os.path.join(output_dir, "model_performance_report.md")
    with open(report_path, 'w') as f:
        f.writelines(report)

    print(f"Report saved to {report_path}")

# Example usage (summary and results would be passed into this function)
summary = {
    "llama3:8b": {
        "code": {"avg_latency": 3.17, "avg_memory_kb": 0.00528, "avg_score": 0.05, "tokens_per_second": 2.80},
        "reasoning": {"avg_latency": 2.03, "avg_memory_kb": 0.00472, "avg_score": 0.09, "tokens_per_second": 2.78},
        "qa": {"avg_latency": 0.35, "avg_memory_kb": 0.0515, "avg_score": 0.20, "tokens_per_second": 1.86},
        "summarization": {"avg_latency": 1.35, "avg_memory_kb": 0.00505, "avg_score": 0.08, "tokens_per_second": 3.34}
    },
    "mistral": {
        "code": {"avg_latency": 1.82, "avg_memory_kb": 0.0498, "avg_score": 0.04, "tokens_per_second": 4.33},
        "reasoning": {"avg_latency": 3.81, "avg_memory_kb": 0.00475, "avg_score": 0.13, "tokens_per_second": 5.22},
        "qa": {"avg_latency": 0.32, "avg_memory_kb": 0.00485, "avg_score": 0.22, "tokens_per_second": 4.63},
        "summarization": {"avg_latency": 0.52, "avg_memory_kb": 0.00500, "avg_score": 0.07, "tokens_per_second": 5.83}
    },
    "phi": {
        "code": {"avg_latency": 0.77, "avg_memory_kb": 0.00515, "avg_score": 0.04, "tokens_per_second": 7.00},
        "reasoning": {"avg_latency": 0.00, "avg_memory_kb": 0.00505, "avg_score": 0.06, "tokens_per_second": 11.29},
        "qa": {"avg_latency": 0.24, "avg_memory_kb": 0.00682, "avg_score": 0.21, "tokens_per_second": 9.92},
        "summarization": {"avg_latency": 0.92, "avg_memory_kb": 0.00490, "avg_score": 0.08, "tokens_per_second": 12.21}
    }
}

# Generate the report
generate_report(summary)
