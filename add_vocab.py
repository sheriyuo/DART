#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extend a local Hugging Face model tokenizer by adding custom tokens.

Usage:
    python extend_tokenizer.py \
        --model_path /path/to/original/model \
        --save_path /path/to/save/extended/model \
        --add_tags

Example:
    python extend_tokenizer.py \
        --model_path ./DeepSeek-R1-0528-Qwen3-8B \
        --save_path ./DeepSeek-R1-0528-Qwen3-8B-AddTags2
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_and_extend_model(model_path, new_tokens=None):
    """
    Load a local Hugging Face model and extend its tokenizer vocabulary.
    Initialize <Analyze> and </Analyze> embeddings using <think> and </think>.

    Args:
        model_path (str): Path to the local model directory
        new_tokens (list): List of new tokens to add to the tokenizer
    """
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    original_vocab_size = tokenizer.vocab_size
    print(f"Original vocabulary size: {original_vocab_size}")

    def load_and_extend_model(new_tokens=None):
        if new_tokens:
            num_added_tokens = tokenizer.add_tokens(new_tokens)
            print(f"Added {num_added_tokens} new tokens")

            if num_added_tokens > 0:
                model.resize_token_embeddings(len(tokenizer))
                print(f"Resized embeddings to new vocab size: {len(tokenizer)}")
                
                # ================== 新增：Smart Initialization ==================
                print("🚀 Initializing new token embeddings with semantic proxies...")
                
                input_embeddings = model.get_input_embeddings().weight.data
                input_std = input_embeddings.std().item()
                output_embeddings = model.get_output_embeddings().weight.data
                output_std = output_embeddings.std().item()

                noise_scale = 0.02
                
                # 定义初始化映射策略
                init_map = {
                    "<search>": "search", 
                    "</search>": "end",   # 或者用 "finish", "done" 甚至 symbol like "}"
                    "<information>": "information",
                    "</information>": "finish",   # 或者用 "finish", "done" 甚至 symbol like "}"
                    "<sql>": "sql",
                    "</sql>": "end",   # 或者用 "finish", "done" 甚至 symbol like "}"
                    "<observation>": "observation",
                    "</observation>": "finish",   # 或者用 "finish", "done" 甚至 symbol like "}"
                }
                
                for new_tok, ref_word in init_map.items():
                    if new_tok not in new_tokens: 
                        continue
                    
                    # 获取新 token 的 ID
                    new_id = tokenizer.convert_tokens_to_ids(new_tok)
                    
                    # 获取参考词的 ID (取第一个 token，防止分词成多个)
                    ref_ids = tokenizer.encode(ref_word, add_special_tokens=False)
                    if not ref_ids:
                        print(f"⚠️ Warning: Reference word '{ref_word}' not found in vocab.")
                        continue
                    ref_id = ref_ids[0]
                    
                    print(f"   Mapping '{ref_word}' (id:{ref_id}) -> '{new_tok}' (id:{new_id})")
                    
                    # 复制 Input Embedding
                    ref_emb = input_embeddings[ref_id]
                    noise = torch.randn_like(ref_emb) * input_std * noise_scale
                    input_embeddings[new_id] = ref_emb + noise
                    
                    # 复制 Output Embedding (lm_head)
                    ref_emb = output_embeddings[ref_id]
                    noise = torch.randn_like(ref_emb) * output_std * noise_scale
                    output_embeddings[new_id] = ref_emb + noise
        return model, tokenizer

    model, tokenizer = load_and_extend_model(new_tokens)
    print(f"Resized embeddings to new vocab size: {len(tokenizer)}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Extend tokenizer and model embeddings with new tokens."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local base model (e.g., ./DeepSeek-R1-0528-Qwen3-8B)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the extended model/tokenizer (e.g., ./DeepSeek-R1-0528-Qwen3-8B-AddTags2)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    # new_tokens = [
    #     "<Analyze>",
    #     "</Analyze>",
    #     "<Understand>",
    #     "</Understand>",
    #     "<Code>",
    #     "</Code>",
    #     "<Execute>",
    #     "</Execute>",
    #     "<Answer>",
    #     "</Answer>",
    # ]
    # new_tokens = ["<search>", "</search>", "<information>", "</information>"] # search r1
    new_tokens = ["<sql>", "</sql>", "<observation>", "</observation>"]  # sql
    print(f"🔹 Adding default tokens: {new_tokens}")

    model, tokenizer = load_and_extend_model(args.model_path, new_tokens)

    sample_text = (
        "<sql>\nTo determine the count of customers supported by Steve Johnson, we need to.\n</sql>\n"
        "<observation>\n```python\n```\n</observation>\n"
    )
    encoded = tokenizer.encode(sample_text, return_tensors="pt")
    decoded = [tokenizer.decode(x) for x in encoded[0]]

    print("\nample text encoding test:")
    print(f"Input: {sample_text}")
    print(f"Encoded tensor shape: {encoded.shape}")
    print(f"Decoded tokens: {decoded[:20]} ...")  # show first 20 tokens only

    os.makedirs(args.save_path, exist_ok=True)
    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)
    print(f"\nExtended model & tokenizer saved to: {args.save_path}")


if __name__ == "__main__":
    main()