import os
import time
from benchmark import LLMBenchmark  
#from visualization import create_visualizations  
from tasks import load_all_benchmarks 
#from report import generate_report

def main():

    # Creat results directory
    os.makedirs("results",exist_ok=True)
    # Define models to benchmark
    #models = ['phi', 'mistral', 'llama3:8b']
    models = ['phi']

    print ("Starting LLM benchmarking")
    start_time = time.time()

    # Load all benchmark tasks
    tasks = load_all_benchmarks()

    # Initialize the LLMBenchmark class with models and tasks
    benchmark = LLMBenchmark(models, tasks)

    # Run benchmarks and collect results
    results = benchmark.run_benchmarks()

    # Calculate summary statistics for all models and tasks
    summary = benchmark.get_summary_statistics()

    # Generate visualizations based on the summary statistics
    #create_visualizations(summary, results_dir="benchmark_results")

    # Generate report
    #generate_report(summary, results)

    # Print completion message and results location
    print(f"Benchmarking completed successfully!")
    print(f"Results saved in: benchmark_results/")

if __name__ == "__main__":
    main()

