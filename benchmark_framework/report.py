import json
import os

def generate_report(summary, all_results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    report = []

    # Title and Introduction
    report.append("# Local LLM Benchmarking on Diversified Problem Sets Using Ollama\n")
    report.append("{}: {}\n{}: {}\n{}: {}\n{}: {}\n".format(
        "Jiaqi Guo", "",
        "Zijun Fu", "",
        "Zishun Shen", "",
        "Ziming Zhang", ""
    ))
    report.append("\n")

    # Project Overview
    report.append("## 1 Introduction\n")
    report.appned(
        """
        Ollama ... Light weight ... Local machine\n
        """
    )
    report.append("\n")

    # Background and Motovation
    report.append("## 2 Background\n")
    report.append("Key benchmarks & models\n")
    report.append("\n")

    # Technical Design
    report.append("## 3 Technical Design\n")
    report.append("### 3.1 Benchmark Datasets\n")
    report.append("\n")
    report.append("### 3.2 Models evaluated\n")
    report.append("\n")
    report.append("### 3.3 Metrics tracked\n")
    report.append("\n")
    report.append("### 3.4 Benchmark Workflow\n")
    report.append(
        """
        benchmark_framework/\n
        ├── benchmark.py         # Core benchmarking engine\n
        ├── tasks.py              # Task definitions and loader\n
        └── visualization.py      # Benchmark result visualizations\n
        data/\n
        ├── qa_benchmark.json\n
        ├── code_benchmark.json\n
        ├── reasoning_benchmark.json\n
        └── summarization_benchmark.json\n
        results/\n
        └── (generated benchmark results)\n
        run_benchmark.py            # Main script to run benchmarks\n
        """
    )
    report.append("\n")
    report.append("### 3.5 UI Design\n")
    report.append("\n")

    # Challenges & Learning
    report.append("## 4 Learning Outcomes\n")
    report.append("What challenges did you met and what did your team learn after overcoming them?\n")
    report.append("\n")

    # Benchmark Results
    report.append("## 5 Results and Evaluation\n")

    # Executive Summary with Key Findings
    report.append("### 5.1 Executive Summary\n")
    report.append(
        """
        This report analyzes the performance in score, latency, memory usage, and tokens per second of different models (Llama3:8b, Mistral, Phi) across various tasks.\n
        """
    )
    report.append("Key findings and takeaways:\n")
    for model in summary.keys():
        for task in summary[model].keys():
            avg_latency = summary[model][task]["avg_latency"]
            avg_memory_kb = summary[model][task]["avg_memory_kb"]
            avg_score = summary[model][task]["avg_score"]
            tokens_per_second = summary[model][task]["tokens_per_second"]
            report.append(
                f"""
                - **{model} - {task}**: 
                Avg Latency: {avg_latency:.2f} s, 
                Avg Memory Usage: {avg_memory_kb:.2f} KB, 
                Avg Score: {avg_score:.2f}, 
                Tokens/sec: {tokens_per_second:.2f}\n
                """
            )
    report.append("\n")

    # Model Rankings
    report.append("### 5.2 Model Rankings\n")
    # Add model rankings based on performance (avg_score)
    model_ranking = []
    for model in summary.keys():
        avg_score = sum([summary[model][task]["avg_score"] for task in summary[model]]) / len(summary[model])
        model_rankings.append((model, avg_score))
    model_rankings = sorted(model_rankings, key=lambda x: x[1], reverse=True)
    report.append("Model Rankings based on average score across all tasks:\n")
    for model, avg_score in model_rankings:
        report.append(f"- **{model}**: Avg Score: {avg_score:.2f}\n")
    report.append("\n")

    # Task-specific Performance Analysis
    report.append("### 5.3 Task-specific Performance Analysis\n")
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
    report.append("### 5.4 Performance vs. Speed Trade-offs\n")
    report.append("![Performance vs. Latency Scatter Plot](performance_vs_latency_scatter.png)\n")
    report.append("\n")

    # Resource Usage Analysis
    report.append("### 5.5 Resource Usage Analysis\n")
    report.append("![Memory Usage Comparison](memory_usage_comparison.png)\n")
    report.append("\n")

    # Example Responses
    report.append("## 5.6 Example Responses\n")
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
    report.append("## 6 Conclusions and Future Works\n")
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
