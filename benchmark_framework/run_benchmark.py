import os
import time
from benchmark import LLMBenchmark  
from visualization import create_visualizations  
from tasks import load_all_benchmarks 
#from report import generate_report

#run 10 times and average the results
def average_summaries(summary_list):
    from collections import defaultdict

    avg_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    count_summary = defaultdict(lambda: defaultdict(int))

    for summary in summary_list:
        for model, tasks in summary.items():
            for task, metrics in tasks.items():
                for metric, value in metrics.items():
                    if value is not None:
                        avg_summary[model][task][metric] += value
                        count_summary[model][task] += 1

    # Final average
    for model in avg_summary:
        for task in avg_summary[model]:
            for metric in avg_summary[model][task]:
                avg_summary[model][task][metric] /= count_summary[model][task]

    return avg_summary


def main():
    # Creat results directory
    os.makedirs("results", exist_ok=True)
    # Define models to benchmark
    models = ['phi', 'mistral', 'llama3:8b']
    # Load all benchmark tasks
    tasks = load_all_benchmarks()

    summary_list = []
    num_runs = 10

    print("Starting repeated benchmarking...")
    for i in range(num_runs):
        print(f"\n Run {i+1}/{num_runs}")
        # Initialize the LLMBenchmark class with models and tasks
        benchmark = LLMBenchmark(models, tasks)
        # Run benchmarks and collect results
        benchmark.run_benchmarks()
        # Calculate summary statistics for all models and tasks
        summary = benchmark.get_summary_statistics()
        summary_list.append(summary)

    # Calculate average summary across runs
    avg_summary = average_summaries(summary_list)

    # Create final visualizations
    create_visualizations(avg_summary, results_dir="benchmark_results")

    # Generate report
    #generate_report(summary, results)
    
    print(" Benchmarking completed with averaged results.")
    print(" Results saved in: benchmark_results/")


if __name__ == "__main__":
    main()
