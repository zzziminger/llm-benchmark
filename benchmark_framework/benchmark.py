import time
import json
import os
from pathlib import Path
import psutil
from difflib import SequenceMatcher
import ollama
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class LLMBenchmark:
    def __init__(self,models,tasks):
        self.models = models
        self.tasks = tasks
        self.results = {}
        os.makedirs('results', exist_ok=True)



    def run_benchmarks(self):
        for model in self.models:
            print(f"Benchmarking {model}...")
            self.results[model] = {}

            for task_name, task_data in self.tasks.items():
                print(f" Running task: {task_name}")
                self.results[model][task_name] = self.benchmark_task(model, task_data)

                self._save_results(model, task_name, self.results[model][task_name])

        return self.results

    def benchmark_task(self, model, task_data):
        results = []
        error_log = []

        print(f"\n{'='*88}")
        print(f" Testing model: {model}")
        print(f"{'='*88}\n")

        for i, item in enumerate(task_data):
            prompt = item['prompt']
            ground_truth = item.get('ground_truth', None)

            print(f"\n Task {i+1}/{len(task_data)}")
            print(f"\n Prompt: \n{'-'*40}\n{prompt}\n{'-'*40}")

            if ground_truth:
                print(f" Expected:{ground_truth}")

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024*1024)
            try: 
                print(" Model thinking", end="", flush = True)
                thinking_thread = threading.Thread(target=self._thinking_animation)
                thinking_thread.daemon = True
                thinking_thread.start()

                try: 
                    response = ollama.generate(model = model, prompt= prompt, options={"timeout":20})
                except Exception as api_err:
                    raise RuntimeError(f"Ollama API call failed: {str(api_err)}")

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

                self._stop_thinking = True
                thinking_thread.join()
        
                latency = end_time - start_time
                memory_increase = max(0, end_memory - start_memory)

                output = response.get("response","").strip()
                if len(output) == 0:
                    raise ValueError("Model returned an empty response.")

                print(f" Response: \n{'-'*40}\n{response['response']}\n{'-'*40}")

                result = {
                    "prompt_id": i,
                    'prompt': prompt,
                    'response': response['response'],
                    'latency': latency,
                    'memory_usage': memory_increase,
                    'tokens_generated': len(response['response'].split()),
                    "tokens_per_second": len(response['response'].split()) / latency if latency > 0 else 0
                    }

                if ground_truth is not None:
                    score = self.evaluate(response['response'], ground_truth,task_type=item.
                                      get('type','general'))
                    result['score'] = score
                    result['ground_truth'] = ground_truth

                    score_str = f"{score: .2f}" if isinstance(score, float) else str(score)
                    emoji = "✅" if score > 0.7 else "❕" if score > 0.3 else "❌" 
                    print(f"{emoji} Score: {score_str}")
                results.append(result)
                    
            except MemoryError:
                self._stop_thinking = True
                thinking_thread.join()
                print("\n❌ MemoryError: Not enough memory to complete the task.")
                error_log.append({
                    "prompt_id":i,
                    "score":0,
                    "reason": "MemoryError - Task esceeded available memory",
                    "actual":""
                })

            except RuntimeError as e:
                self._stop_thinking = True
                thinking_thread.join()
                print(f"\n❌ RuntimeError: {str(e)}")
                error_log.append({
                    "prompt_id":i,
                    "score":0,
                    "reason": str(e),
                    "actual":""
                })

            except Exception as e:
                self._stop_thinking = True
                thinking_thread.join()
                print(f"\n❌ Unexpected Error: {str(e)}")
                error_log.append({
                    "prompt_id":i,
                    "score":0,
                    "reason": f"Unhandled error: {str(e)}",
                    "actual":""
                }) 
                print("\n" + "-" *80)
                time.sleep(0.5) 

        if error_log:
            os.makedirs("results", exist_ok=True)
            with open("results/error_log.txt", "w") as f:
                for err in error_log:
                    f.write(f"Prompt ID: {err['prompt_id']}\n")
                    f.write(f"Reason: {err['reason']}\n")
                    f.write(f"Model Answer: {err['actual']}\n")
                    f.write(f"Score: {err['score']}\n")
                    f.write("-" * 60 + "\n")     

        return results 



    def _thinking_animation(self):
        self._stop_thinking = False
        animation = "/-\\"
        idx = 0
        while not self._stop_thinking:
            print(f"\r🧠 Model thinking {animation[idx % len(animation)]}", end="",flush = True)
            idx += 1
            time.sleep(0.1)

    def evaluate(self, response, ground_truth, task_type):
        
        if task_type == "qa":
            return int(ground_truth.lower() in response.lower())
        
        elif task_type == "code":
            return self.code_similarity(response, ground_truth)
        
        elif task_type == "summarization":
            return self.text_similarity(response, ground_truth)
        
        elif task_type == "reasoning":
            return self.text_overlap(response, ground_truth)
        
        else:
            return self.text_similarity(response,ground_truth)



    def text_similarity(self, a, b):
        words1 = set(a.lower().split())
        words2 = set(b.lower().split())

        if not words1 or not words2:
            return 0.0
        overlap = len(words1.intersection(words2))
        return overlap / max(len(words1),len(words2))

    def text_overlap(self, a, b):
        a_short = self.shorten_response(a)
        a_words = set(a_short.lower().split())
        b_words = set(b.lower().split())

        if not a_words or not b_words:
            return 0.0
        
        overlap = len(a_words.intersection(b_words))
        precision = overlap / len(a_words) if a else 0
        recall = overlap / len(b_words) if b_words else 0

        if precision + recall == 0:
            return 0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        concept_overlap = self.conceptual_overlap(a, b)

        return f1_score * (1 + concept_overlap)
    
    def conceptual_overlap(self, a, b):
        key_concepts_a = set(self.extract_key_concepts(a))
        key_concepts_b = set(self.extract_key_concepts(b))

        if not key_concepts_a or not key_concepts_b:
            return 0.0
        
        overlap = len(key_concepts_a.intersection(key_concepts_b))
        
        return overlap / max(len(key_concepts_a), len(key_concepts_b))
    
    def extract_key_concepts(self, text):
        common_stopwords = set([
        'is', 'the', 'a', 'and', 'of', 'or', 'in', 'to', 'on', 'for', 'with', 'as', 'by', 'that'])
        words = text.lower().split()
        return [word for word in words if word not in common_stopwords]


    def code_similarity(self, a, b):
        a_clean = ''.join(a.split())
        b_clean = ''.join(b.split())

        # Use TfidfVectorizer to convert code into vectors and calculate cosine similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([a_clean, b_clean])

        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        return similarity_matrix[0][0]
    
    def shorten_response(self, response, max_sentences=3, max_words=100):
    # Take the first few sentences, then truncate to max_words
        sentences = response.strip().split('.')
        selected = '.'.join(sentences[:max_sentences])
        words = selected.split()
        return ' '.join(words[:max_words])


    def _display_interaction(self, prompt, response, score=None, latency=None):
        separator = "─" * 60
        print(f"\n\033[95m{separator}")
        print(f"\033[1m🧠 Prompt:\033[0m\n{prompt}")
        print(f"\n\033[92m📤 Model Response:\033[0m\n{response}")
        if score is not None:
            print(f"\n\033[96m✅ Score:\033[0m {score:.2f}")
        if latency is not None:
            print(f"\033[93m⏱️ Latency:\033[0m {latency:.2f}s")
        print(f"\033[95m{separator}\033[0m")

    def _save_results(self, model, task_name, data):
        results_folder = 'results'
        os.makedirs(results_folder, exist_ok=True)
        filename = Path(results_folder) / f"{model.replace(':', '_')}_{task_name}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def get_summary_statistics(self):
        summary = {}
        for model, tasks in self.results.items():
            summary[model] = {}
            for task_name, results in tasks.items():
                latencies = [r["latency"] for r in results if "latency" in r]
                memories = [r["memory_usage"] for r in results if "memory_usage" in r]
                scores = [r["score"] for r in results if "score" in r]
                summary[model][task_name] = {
                    "avg_latency": sum(latencies) / len(latencies),
                    "avg_memory_kb": sum(memories) / len(memories),
                    "avg_score": sum(scores) / len(scores) if scores else None,
                }
        return summary


