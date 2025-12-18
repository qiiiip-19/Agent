import os
import json
import re
import time
import argparse
import requests
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

from bs4 import BeautifulSoup
from datasets import Dataset
from zai import ZhipuAiClient
import wikipedia

# ================= Configuration =================
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

SUMMARY_PROMPT_TEMPLATE = '''## Task Description:
Given the search query and the content of the searched webpage, extract relevant information into a summary paragraph.

## Inputs:
[Search Query]
{search_query}

[Webpage Content]
{document}

## Output Format:
[Exacted Content]: If related, output summary; if not, output "None".
'''

# Environment Variables
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
ZHIPU_MODEL_NAME = os.environ.get("ZHIPU_MODEL")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

# Initialize Clients
client = ZhipuAiClient(api_key=ZHIPU_API_KEY)
session = requests.Session()
session.headers.update(headers)
wikipedia.set_lang('en')
wikipedia._http = session

# ================= URL Logging =================
url_log_lock = threading.Lock()
url_log: List[Dict[str, Any]] = []
url_stats = {"total": 0, "fail": 0}


def log_url_access(url: str, success: bool, source: str) -> None:
    """Thread-safe logging of URL access outcome."""
    global url_log, url_stats
    with url_log_lock:
        url_stats["total"] += 1
        if not success:
            url_stats["fail"] += 1
        url_log.append(
            {
                "url": url,
                "success": success,
                "source": source,
                "timestamp": time.time(),
            }
        )

# ================= Helper Tools (Search & Scrape) =================

def extract_text_from_url(url: str) -> str:
    """Fetches and parses text from a URL, with special handling for Wikipedia."""
    try:
        if "wikipedia.org" in url:
            # Try up to 3 times for Wikipedia
            for _ in range(3):
                try:
                    page_title = requests.utils.unquote(url.split('/')[-1])
                    page = wikipedia.page(page_title, auto_suggest=False)
                    # Simple cleanup
                    content = page.content.replace('\n\n', '\n')
                    # Remove references/notes sections
                    log_url_access(url, True, "wikipedia")
                    return content.split('== References ==')[0].split("== Notes ==")[0].strip()
                except Exception:
                    time.sleep(1)
            log_url_access(url, False, "wikipedia")
            return "None"
        else:
            # General Web Scraping
            for _ in range(3):
                try:
                    response = session.get(url, timeout=15)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'lxml')
                        # Extract text with newline separator
                        text = soup.get_text(separator='\n', strip=True)
                        # Filter out very short lines (often menu items/nav)
                        lines = [line for line in text.split("\n") if len(line.split()) >= 3]
                        log_url_access(url, True, "webpage")
                        return " ".join(lines)
                except Exception:
                    time.sleep(2)
            log_url_access(url, False, "webpage")
            print(f"[Warn] Failed to access url: {url}")
            return "None"
    except Exception as e:
        log_url_access(url, False, "webpage")
        print(f"[Error] Extracting text failed: {e}")
        return "None"

def serper_web_search(query: str, api_key: str) -> dict:
    """Executes a Google search via Serper API."""
    print(f"[Search] model requests to search: {query[:100]}")

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": 5}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        # Log Serper API call as a kind of "URL access"
        log_url_access(url, True, "serper_api")
        return resp.json()
    except Exception as e:
        log_url_access(url, False, "serper_api")
        print(f"[Error] Serper request failed: {e}")
        return {}

def extract_relevant_info(search_results: dict) -> List[dict]:
    """Parses Serper JSON to get titles and URLs."""
    return [{'title': r.get('title'), 'url': r.get('link')} 
            for r in search_results.get('organic', []) if r.get('link')]

# ================= LLM Interaction =================

def parse_qwen_prompt(prompt: str) -> List[Dict]:
    """
    Manually parses a raw string prompt with <|im_start|> tags into a messages list.
    Necessary because we are appending raw text to history during the loop.
    """
    messages = []
    pattern = r'<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>'
    matches = re.findall(pattern, prompt, re.DOTALL)
    
    for role, content in matches:
        if role in ['system', 'user']:
            messages.append({"role": role, "content": content.strip()})
        # Note: 'assistant' parts are history, usually handled by appending to the last user message 
        # or kept as context if the API supports it. 
        # In this specific logic, we might be collapsing history into the prompt.
        
    # Handle the tail of the prompt (the new part to generate)
    last_end = prompt.rfind('<|im_end|>')
    if last_end != -1:
        remaining = prompt[last_end + len('<|im_end|>'):].strip()
        if remaining:
             # Append remaining text to the last user message or create a new one
            if messages and messages[-1]['role'] == 'user':
                messages[-1]['content'] += "\n" + remaining
            else:
                messages.append({"role": "user", "content": remaining})
    elif not messages:
        # Fallback
        messages.append({"role": "user", "content": prompt})
        
    return messages

def call_llm(prompt: str, model: str, temperature: float = 0.0, stop: Optional[List[str]] = None) -> tuple[str, str]:
    """
    Wrapper for ZhipuAI API call.
    Returns: (generated_text, finish_reason)
    """
    try:
        messages = parse_qwen_prompt(prompt) if "<|im_start|>" in prompt else [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048 if "Task Description" in prompt else 512, # Long context for summary, short for reasoning
            temperature=temperature,
            stop=stop
        )
        content = response.choices[0].message.content
        reason = response.choices[0].finish_reason
        return content, reason
    except Exception as e:
        print(f"[Error] LLM Call failed: {e}")
        return "", "error"

# ================= Core Logic (ReAct Loop) =================

def process_single_step(output_text: str, stop_reason: str, current_data: dict, step_idx: int) -> tuple[dict, str]:
    """
    Analyzes the LLM output to decide:
    1. Finished (Answer found)
    2. Continue (Need to search)
    3. Error/Max steps reached
    """
    prompt = current_data.get("chat_prompt", "") # Note: In current logic, prompt is accumulated
    answer_gt = current_data["answer"]
    question = current_data["question"]
    gen_text_store = current_data.get("gen_text_store", "")

    # Case 0: Empty or Error
    if not output_text:
        return {
            "question": question, "answer": answer_gt,
            "generated_text": output_text, "stop_reason_final": "empty_output",
            "pred_ans": "I don't know."
        }, "finished"

    # Case 1: Max Retrieval Steps Reached (Force Stop)
    if step_idx >= 8: 
        return {
            "question": question, "answer": answer_gt,
            "stop_reason_final": "max_steps_reached",
            "pred_ans": "I don't know.",
            "gen_text_store": gen_text_store + output_text
        }, "finished"

    # Case 2: Final Answer Found (<answer> tags)
    if "<answer>" in output_text:
        print(f"[Answer] final answer found: {output_text[:100]}")

        if "</answer>" in output_text:
            pred_ans = output_text.split("<answer>")[-1].split("</answer>")[0]
            full_gen = output_text
        else:
            # Model stopped generating but started the tag
            pred_ans = output_text.split("<answer>")[-1].strip()
            full_gen = output_text + "</answer>"
            
        return {
            "question": question, "answer": answer_gt,
            "pred_ans": pred_ans,
            "stop_reason_final": "finished",
            "gen_text_store": gen_text_store + full_gen,
        }, "finished"

    # Case 3: Need Search (<|begin_of_query|>)
    elif "<|begin_of_query|>" in output_text:
        print(f"[Need search] read <|begin_of_query|>: {output_text[:100]}")

        # Extract Query
        if "<|end_of_query|>" in output_text:
            query = output_text.split("<|begin_of_query|>")[-1].split("<|end_of_query|>")[0]
        else:
            query = output_text.split("<|begin_of_query|>")[-1].strip()
        
        query = query.replace('"', "").replace("\t", " ").strip()
        
        if query:
            # 1. Execute Search
            search_results = serper_web_search(query, SERPER_API_KEY) if SERPER_API_KEY else {}
            extracted_info = extract_relevant_info(search_results)
            
            doc_summary_content = "None"
            
            # 2. Browse & Summarize (Top 3 results)
            for info in extracted_info[:3]:
                raw_text = extract_text_from_url(info['url'])
                if not raw_text or raw_text == "None": continue
                
                # Truncate to avoid context overflow before summarization
                summary_input = SUMMARY_PROMPT_TEMPLATE.format(
                    search_query=query, 
                    document=raw_text[:35000]
                )
                
                # Call LLM to summarize the page
                summary_res, _ = call_llm(summary_input, ZHIPU_MODEL_NAME)
                
                if "[Exacted Content]" in summary_res:
                    extracted = summary_res.split("[Exacted Content]")[-1].lstrip(":").strip()
                    if "none" not in extracted.lower():
                        doc_summary_content = extracted
                        break # Found useful info, stop browsing
            
            # 3. Construct Context for Next Step
            # Append the Action (Query) + Observation (Documents) to the history
            new_context_suffix = f"{output_text.strip()}<|end_of_query|>\n\n<|begin_of_documents|>\n{doc_summary_content}\n<|end_of_documents|>\n\n"
            
            return {
                "chat_prompt": prompt + new_context_suffix, # Accumulate context
                "answer": answer_gt,
                "question": question,
                "gen_text_store": gen_text_store + new_context_suffix,
            }, "continued"
        else:
            # Malformed query
            return {
                "question": question, "answer": answer_gt,
                "stop_reason_final": "query_parse_error",
                "pred_ans": "I don't know."
            }, "finished"

    # Case 4: Model rambling or unknown state
    else:
        return {
            "question": question, "answer": answer_gt,
            "generated_text": output_text, "stop_reason_final": "unknown_stop",
            "pred_ans": "I don't know."
        }, "finished"

# ================= Data Processing =================

def format_prompt(example, prompt_type="v3"):
    """Applies the specific system prompt template."""
    question = example["question"]
    
    # Templates (Simplified for brevity, verify against original requirements)
    templates = {
        "v0": "You are a helpful assistant.\n... <|begin_of_query|> ... <|end_of_query|> ...",
        "v2": "You are a helpful assistant. Given a **Judgement question**...",
        "v3": """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format is "<think>...</think> <answer>...</answer>".
During thinking, you can search using "<|begin_of_query|> keywords <|end_of_query|>".
"""
    }
    
    system_content = templates.get(prompt_type, templates["v3"])
    # Manual construction of Qwen format
    example["chat_prompt"] = f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>"
    return example

# ================= Main Execution =================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, required=True, help="Input .jsonl file")
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--end_sample", type=int, default=10000)
    parser.add_argument("--gpu_id", type=str, default="0") # Kept for compatibility
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--prompt_type", type=str, default="v3")
    parser.add_argument("--model_path", type=str, default="default") # For naming
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id # Not strictly needed for API but good practice
    
    if not ZHIPU_MODEL_NAME or not ZHIPU_API_KEY:
        raise ValueError("Environment variables ZHIPU_MODEL or ZHIPU_API_KEY are missing.")

    # Generate Output Filename
    model_suffix = args.model_path.split("/")[-1] if args.model_path else "unknown"
    output_file = f"{args.src_file.replace('.jsonl', '')}-{model_suffix}.jsonl"
    print(f"Outputting to: {output_file}")

    # Load Data
    with open(args.src_file, "r") as f:
        all_lines = f.readlines()
        
    data_slice = [json.loads(line) for i, line in enumerate(all_lines) 
                  if args.start_sample <= i < args.end_sample]
    
    print(f"Processing {len(data_slice)} samples...")

    # Processing Chunks
    chunk_size = 5 # Small chunk size for debugging stability
    for i in range(0, len(data_slice), chunk_size):
        chunk_data = data_slice[i : i + chunk_size]
        
        # 1. Initial Prompt Formatting
        ds = Dataset.from_list(chunk_data)
        ds = ds.map(lambda x: format_prompt(x, args.prompt_type), num_proc=4)
        
        # Track active items (some finish early, some continue)
        active_items = [x for x in ds] 
        stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        
        finished_results = []
        
        # 2. Multi-turn Loop (Max 16 turns)
        for step in range(16):
            if not active_items:
                break
                
            print(f"--- Step {step}: Processing {len(active_items)} items ---")
            
            # Batch Generation (Parallel API Calls)
            step_outputs = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(call_llm, item["chat_prompt"], ZHIPU_MODEL_NAME, args.temp, stop_tokens): item 
                    for item in active_items
                }
                
                next_round_items = []
                
                # Process Results as they arrive
                for future in as_completed(futures):
                    original_item = futures[future]
                    gen_text, stop_reason = future.result()
                    
                    # Print preview for debug
                    if len(gen_text) > 0:
                        print(f"[Debug] Gen preview: {gen_text[:100].replace(chr(10), ' ')}...")
                    
                    # Decide Next Step
                    result_data, status = process_single_step(gen_text, stop_reason, original_item, step)
                    
                    if status == "finished":
                        finished_results.append(result_data)
                    else:
                        next_round_items.append(result_data)
            
            # Update active list for next iteration
            active_items = next_round_items
            
            # Save intermediate results (Good for long runs)
            if finished_results:
                with open(output_file, "a") as f_out:
                    for res in finished_results:
                        f_out.write(json.dumps(res) + "\n")
                finished_results = [] # Clear buffer

    # Write URL access log
    model_suffix = args.model_path.split("/")[-1] if args.model_path else "unknown"
    url_log_file = f"{args.src_file.replace('.jsonl', '')}-{model_suffix}-url_log.jsonl"
    if url_log:
        with open(url_log_file, "w") as f_log:
            for item in url_log:
                f_log.write(json.dumps(item) + "\n")

    # Print failure ratio
    total = url_stats["total"]
    fail = url_stats["fail"]
    if total > 0:
        ratio = fail / total
        print(f"=== URL Access Stats ===")
        print(f"Total accesses: {total}, Failures: {fail}, Failure ratio: {ratio:.2%}")
    else:
        print("=== URL Access Stats ===")
        print("No URL accesses performed.")

    print("=== Processing Complete ===")

if __name__ == "__main__":
    main()