# DART: Disentangled Action Reasoning Tuning

Unofficial open-source implementation based on the paper [Reasoning and Tool-use Compete in Agentic RL: From Quantifying Interference to Disentangled Tuning](https://arxiv.org/abs/2602.00994)


### Core Steps

#### 1. Add Vocabulary Fine-tuning

Use the `add_vocab.py` script to perform vocabulary fine-tuning on the model:

```bash
python add_vocab.py
```

#### 2. Supervised Fine-tuning (SFT)

We use ms-swift for SFT training. Start the SFT training using the provided script:

```bash
bash examples/train/search_r1/sft_3b.sh
```

**Note on SFT Dataset:**
- For fairness in evaluation, the SFT training uses a self-distilled dataset.
- **Download link:** [Google Drive](https://drive.google.com/file/d/1WmbStMw7JbqACohdqdIaHJf2j6sDEVTF/view?usp=sharing)

#### 3. Add LoRA to the Original Model

Use the `add_lora.py` script to add LoRA layers to the original Qwen2 model:

```bash
python3 lora_model/add_lora.py --src <original_model_path> --dst <target_model_path> [optional_args]
```

**Parameter Description:**
- `--src`: Path to the original model (e.g., Qwen2.5-7B)
- `--dst`: Target path to save the model with LoRA layers
- `--lora_r`: LoRA rank, default is 8
- `--lora_alpha`: LoRA scaling factor, default is 16
- `--lora_dropout`: LoRA dropout rate, default is 0.0
- `--dtype`: Data type, default is "auto"

**Example:**
```bash
python3 lora_model/add_lora.py --src /path/to/qwen2.5-7b --dst /path/to/qwen2.5-7b-lora --lora_r 32 --lora_alpha 64
```

#### 2. Copy Model Files to Target Folder

Copy the `modeling_qwen2.py` and `configuration_qwen2.py` files from the `lora_model` directory to the target model folder created in step 1:

```bash
cp lora_model/modeling_qwen2.py lora_model/configuration_qwen2.py <target_model_path>/
```

**Example:**
```bash
cp lora_model/modeling_qwen2.py lora_model/configuration_qwen2.py /path/to/qwen2.5-7b-lora/
```

#### 3. Run the Training Script

Start Search-R1 style training using the provided training script:

```bash
bash examples/train/search_r1/train_7b.sh
```

### Training Script Configuration

The `train_7b.sh` script contains the following key configuration parameters that can be modified as needed:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `model_name` | Path to the training model | `/text2sql/verl-tool/sft_model/Qwen2.5-7B-AddTags/v1-20251225-233050/checkpoint-100-2lora` |
| `train_data` | Path to training data | `/text2sql/verl-tool/data/searchR1_processed_direct/train.parquet` |
| `val_data` | Path to validation data | `/text2sql/verl-tool/data/searchR1_processed_direct/test.parquet` |
| `rl_alg` | RL algorithm (grpo or gae(ppo)) | `grpo` |
| `n_gpus_per_node` | Number of GPUs per node | `8` |
| `n_nodes` | Number of nodes | `1` |
| `total_epochs` | Total number of training epochs | `3` |
| `total_training_steps` | Total number of training steps | `100` |
| `batch_size` | Batch size | `256` |
| `lr` | Learning rate | `1e-5` |
| `kl_loss_coef` | KL loss coefficient | `0.001` |
| `reward_manager` | Reward manager | `search_r1_qa_em` |

**Required Mandatory Configurations:**
The following parameters must be set in the training configuration:
- `actor_rollout_ref.rollout.enforce_eager=True`
- `actor_rollout_ref.rollout.model_impl=transformers`

### Data and Model Preparation

1. **Prepare Original Model**: Ensure you have downloaded the original Qwen2 model (e.g., Qwen2.5-7B)

2. **Prepare Training Data**:
   - Training data should be in Parquet format
   - Data should contain fields such as queries, search results, and answers
   - You can use scripts in the `examples/data_preprocess/` directory to process raw data

3. **Prepare Retrieval Server**:
   - The training script will automatically start the retrieval server
   - Ensure necessary index files and corpus files are available in the `data/QAdataset/` directory

### Monitoring Training Progress

During training, you can monitor the progress through:

1. **Log Output**: The training script outputs detailed logs including loss, reward, KL divergence, and other metrics

2. **TensorBoard**: If TensorBoard is enabled, you can view training curves with:
```bash
tensorboard --logdir logs/
```

### Custom vllm Installation

This custom version is installed to ensure the proper functioning of the router module in the DART framework, which enables the accurate routing of reasoning and tool-use tokens to their corresponding parameter subspaces and prevents interference between different capabilities.
Please follow these steps to install it:

```bash
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

## Project Structure

```
├── lora_model/              # LoRA-related code and model files
│   ├── add_lora.py          # Script to add LoRA to the original model
│   ├── modeling_qwen2.py    # Qwen2 model implementation with LoRA
│   └── configuration_qwen2.py # Qwen2 model configuration file
├── examples/                # Example code
│   ├── train/               # Training examples
│   │   └── search_r1/       # Search-R1 training scripts
│   └── eval/                # Evaluation examples
├── vllm/                    # Custom vllm implementation
├── add_vocab.py             # Script for vocabulary fine-tuning
├── sft.sh                   # Script for SFT training
├── verl/                    # verl submodule
└── verl_tool/               # VerlTool core code
```

## Additional Resources

- 📖 [Installation Guide](./assets/docs/install.md)
- ⚡ [Synchronous Rollout Design](./assets/docs/sync_design.md)
- 🔄 [Asynchronous Rollout Design](./assets/docs/asyncRL.md)
- 🛠️ [Tool Server Design](./assets/docs/tool_server.md)
- 🎯 [Training Guide](./assets/docs/training_guide.md)
- 📊 [Evaluation Guide](./assets/docs/evaluation.md)

## Acknowledgements

This repository is a secondary development based on [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool.git) from TIGER-AI-Lab.

## Citation

```bibtex
@article{li2026reasoning,
  title={Reasoning and Tool-use Compete in Agentic RL:From Quantifying Interference to Disentangled Tuning}, 
  author={Yu, Li and Mingyang, Yi and Xiuyu, Li and Ju, Fan and Fuxin, Jiang and Binbin, Chen and Peng, Li and Jie, Song and Tieying, Zhang},
  journal={arXiv preprint arXiv:2602.00994},
  year={2026}
}
```
