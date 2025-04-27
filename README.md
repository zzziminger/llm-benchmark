# 🧪 LLM Benchmarking Project 

In this project, we develop a robust system to benchmark Large Language Models (LLMs) running locally on our laptops. The system will evaluate the performance of three models ('phi', 'mistral', 'llama3:8b') across a range of tasks (Q&A, coding, summarization, reasoning) and display the results visually—featuring an intuitive interface that highlights the model's thought process in real-time.

## 📂 Directory Structure and Description
```
benchmark_framework/
  ├── benchmark.py         # Core benchmarking engine
  ├── tasks.py              # Task definitions and loader
  └── visualization.py      # Benchmark result visualizations
data/
  ├── qa_benchmark.json
  ├── code_benchmark.json
  ├── reasoning_benchmark.json
  └── summarization_benchmark.json
results/
  └── (generated benchmark results)
run_benchmark.py            # Main script to run benchmarks
```

## 🚀 Quick Start


1. Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Pull the LLMs for benchmarking

```bash
ollama pull phi
ollama pull mistral
ollama pull llama3:8b
```

3. Clone the repository
   
```bash
git clone <repository-url>
```

4. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

5. Install dependencies

```bash
pip install scikit-learn psutil ollama pandas matplotlib seaborn
```

6. Run the benchmark

``` bash
python run_benchmark.py
```

