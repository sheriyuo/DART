import fire
from openai import OpenAI
from transformers import AutoTokenizer

def main(
    model_name: str,
    base_url: str,
    test_task: str = "math",
    test_type: str = "chat_completion", # or "completion"
    api_key: str = "sk-proj-1234567890",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_p: float = 1.0,
    n: int = 1,
):
    client = OpenAI(api_key=api_key, base_url=base_url)  # Replace with your local server address
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # get test_task
    if test_task == "math":
        print("Testing math task...")
        system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."
        math_problem = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"

        chat_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": math_problem
            }
        ]
        prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    elif test_task == "search":
        print("Testing search task...")
        system_prompt = "You are a helpful and harmless assistant."
        math_problem = "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: total number of death row inmates in the us?"
        chat_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": math_problem
            }
        ]
        prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    else:
        raise ValueError(f"Unknown test task: {test_task}")
    
            
    if test_type == "chat_completion":
        print(f"Testing {test_task} with {test_type} on model {model_name} at {base_url}", flush=True)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": math_problem
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
        )
        print(completion.choices[0].message.content)
    elif test_type == "completion":
        print(f"Testing {test_task} with {test_type} on model {model_name} at {base_url}", flush=True)
        chat_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": math_problem
            }
        ]
        prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        completion = client.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
        )
        print(completion.choices[0].text)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)

"""
# test math model
python3 eval_service/test/test_api.py --model_name checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6debug/global_step_370/actor/huggingface" --test_task search --test_type chat_completion --base_url http://localhost:5000
"""
