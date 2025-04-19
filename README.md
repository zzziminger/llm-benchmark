# LLM Benchmarking Project 

In this project, we’ll develop a robust system to benchmark Large Language Models (LLMs) running locally on our laptops. The system will evaluate the performance of three models ('phi', 'mistral', 'llama3:8b') across a range of tasks (Q&A, coding, summarization, reasoning) and display the results visually—featuring an intuitive interface that highlights the model's thought process in real-time.

## Directory Structure
llm-benchmark/
├── benchmark_framework/
│   ├── __init__.py
│   ├── benchmark.py     # Core benchmarking engine
│   ├── tasks.py         # Task loaders and definitions
│   ├── visualization.py # Generate plots and visuals
│   └── report.py        # Auto-generate benchmark report
├── data/
│   ├── qa_benchmark.json
│   ├── code_benchmark.json
│   ├── reasoning_benchmark.json
│   └── summarization_benchmark.json
├── results/             # Output directory (created by the code)
└── run_benchmark.py     # Main execution script









