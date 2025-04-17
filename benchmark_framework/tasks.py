import json
import os

def create_qa_benchmark(questions_file):
    """
    Load and format question-answering benchmark tasks.

    Args:
        file_path (str): Path to the JSON file containing QA pairs

    Returns:
        list: List of formatted benchmark tasks
    """
    with open(questions_file, 'r') as file:
        questions = json.load(file)
    
    tasks = []
    for q in questions:
        task = {
            "task_type": "qa",
            "prompt": f"Question: {q['question']}\nAnswer: ",
            "ground_truth": q['answer']
        }
        tasks.append(task)
    
    return tasks

def create_code_benchmark(problems_file):
    """
    Load and format code generation benchmark tasks.

    Args:
        file_path (str): Path to the JSON file containing coding problems

    Returns:
        list: List of formatted benchmark tasks
    """
    with open(problems_file, 'r') as file:
        problems = json.load(file)
    
    tasks = []
    for p in problems:
        task = {
            "task_type": "code",
            "prompt": f"Write a function to solve the following problem: \n\n {p['problem']}\n \n Your solution (in Python): \n",
            "ground_truth": p.get('solution',''),
        }
        tasks.append(task)
    
    return tasks

def create_summarization_benchmark(article_file):
    """
    Load and format text summarization benchmark tasks.

    Args:
        file_path (str): Path to the JSON file containing text to summarize

    Returns:
        list: List of formatted benchmark tasks
    """
    with open(article_file, 'r') as file:
        articles = json.load(file)
    
    tasks = []
    for a in articles:
        task = {
            "task_type": "summarization",
            "prompt": f"TSummarize the following text in a few sentences: \n\n{a['text']} \n\nSummary: ",
            "ground_truth": a.get('summary','')
        }
        tasks.append(task)
    
    return tasks


def create_reasoning_benchmark(contexts_file):
    """
    Load and format reasoning benchmark tasks.

    Args:
        file_path (str): Path to the JSON file containing reasoning problems

    Returns:
        list: List of formatted benchmark tasks
    """
    with open(contexts_file, 'r') as file:
        context = json.load(file)
    
    tasks = []
    for c in context:
        task = {
            "task_type": "reasoning",
            "prompt": f"{p['context']} \n\nQuestion: {p['question']}\n\nAnswer:  ",
            "ground_truth": c['answer']
        }
        tasks.append(task)
    
    return tasks

def load_all_benchmarks(data_dir="data"):
    """
    Load all benchmark tasks from the data directory.

    Args:
        data_dir (str): Path to directory containing benchmark data files

    Returns:
        dict: Dictionary mapping task types to lists of benchmark tasks
    """
    benchmarks = {}

    benchmark_functions = {
        "qa_benchmark.json": create_qa_benchmark,
        "code_benchmark.json": create_code_benchmark,
        "summarization_benchmark.json": create_summarization_benchmark,
        "reasoning_benchmark.json": create_reasoning_benchmark
    }

    for filename, creat_func in benchmark_functions.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            benchmark_name = filename.split('_')[0]
            benchmarks[benchmark_name] = creat_func(filepath)
            print(f"Loaded {benchmark_name} benchmark with {len(benchmarks[benchmark_name])} tasks")
        else:
            print(f"Warning: Benchmark file {filepath} not found")

    return benchmarks 
