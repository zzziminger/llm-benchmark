{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd2b2485",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ui/app.py\n",
    "\n",
    "import streamlit as st\n",
    "from runner import run_benchmark \n",
    "\n",
    "st.set_page_config(page_title=\"LLM Benchmarking Tool\", layout=\"wide\")\n",
    "\n",
    "st.title(\"🧠 LLM Benchmark Playground\")\n",
    "st.markdown(\"Benchmark 3–4 models locally across custom QA, summarization, code, and reasoning tasks.\")\n",
    "\n",
    "# 选择模型\n",
    "models = st.multiselect(\n",
    "    \"Select models to benchmark:\",\n",
    "    options=[\"phi\", \"mistral\", \"llama3:8b\", \"llama3:70b\"],\n",
    "    default=[\"phi\", \"mistral\"]\n",
    ")\n",
    "\n",
    "# 启动按钮\n",
    "if st.button(\"🚀 Run Benchmark\"):\n",
    "    with st.spinner(\"Running...\"):\n",
    "        summary, _ = run_benchmark(models)\n",
    "\n",
    "    st.success(\"✅ Benchmarking completed!\")\n",
    "    st.markdown(\"### 📊 Summary Statistics\")\n",
    "    \n",
    "    for model in summary:\n",
    "        st.markdown(f\"**Model: `{model}`**\")\n",
    "        for task, stats in summary[model].items():\n",
    "            col1, col2, col3 = st.columns(3)\n",
    "            col1.metric(f\"{task} - Latency\", f\"{stats['avg_latency']:.2f}s\")\n",
    "            col2.metric(f\"{task} - Mem Use\", f\"{stats['avg_memory_kb']:.1f} MB\")\n",
    "            col3.metric(f\"{task} - Score\", f\"{stats['avg_score']:.2f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
