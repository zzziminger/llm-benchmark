# LLM Benchmarking Project 

In this project, we’ll develop a robust system to benchmark Large Language Models (LLMs) running locally on our laptops. The system will evaluate the performance of three models ('phi', 'mistral', 'llama3:8b') across a range of tasks (Q&A, coding, summarization, reasoning) and display the results visually—featuring an intuitive interface that highlights the model's thought process in real-time.

## Directory Structure and Description

- **benchmark_framework/**: Contains the core components for the benchmarking system.
  - `benchmark.py`: Implements the benchmarking engine that runs the tests.
  - `tasks.py`: Defines the tasks (QA, code, reasoning, summarization) and handles task loading.
  - `visualization.py`: Generates visual representations (e.g., plots) of the benchmark results.

- **data/**: Stores the benchmark datasets for different task types.
  - `qa_benchmark.json`: Contains data for QA tasks.
  - `code_benchmark.json`: Contains data for code-related tasks.
  - `reasoning_benchmark.json`: Contains data for reasoning tasks.
  - `summarization_benchmark.json`: Contains data for summarization tasks.

- **results/**: This directory is created by the system to store the benchmark results.
  
- **run_benchmark.py**: The main script used to execute the benchmarking process.

## Implementation Guide

### Part 1: Environment Setup

#### Install Ollama
To install Ollama, run the following command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Next, pull some models to use for benchmarking:

```bash
ollama pull phi
ollama pull mistral
ollama pull llama3:8b
```

Now, set up a virtual Python environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

