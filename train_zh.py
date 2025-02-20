import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM

CONFIG = {
    "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    "dataset_name": "minyichen/tw-instruct-500k-cleaned",
    "train_split": "train",
    "response_template": "Response:",
    "max_seq_length": 4096,
    "learning_rate": 5e-6,
    "num_train_epochs": 6,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 10,
    "warmup_steps": 10,
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),
    "logging_steps": 1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "push_to_hub": True,
    "output_dir": "open-s1-mistral-small-24b-zh",
    "dataset_text_field": "text",
    "packing": False,
    "dataset_num_proc": 2,
}


def load_model_and_tokenizer(model_name):
    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_cache=False,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        rope=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


# Prompt Formatting Function
def formatting_prompts_func(examples):
    texts = []
    for input_text, output_result in zip(examples["input"],
                                         examples["output"]):
        prompt = f"{input_text}\n"
        messages = [
            {"role": "system",
             "content": "You are a helpful and harmless assistant. You should think step-by-step."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Response: {output_result}<|eot_id|>"},
        ]
        texts.append(tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        ))
    return {"text": texts}


model, tokenizer = load_model_and_tokenizer(CONFIG["model_name"])
dataset = load_dataset(CONFIG["dataset_name"], split='train').map(formatting_prompts_func, batched=True)
collator = DataCollatorForCompletionOnlyLM(CONFIG["response_template"], tokenizer=tokenizer)

sft_config = SFTConfig(
    max_seq_length=CONFIG["max_seq_length"],
    learning_rate=CONFIG["learning_rate"],
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    save_strategy='epoch',
    warmup_steps=CONFIG["warmup_steps"],
    fp16=CONFIG["fp16"],
    bf16=CONFIG["bf16"],
    logging_steps=CONFIG["logging_steps"],
    weight_decay=CONFIG["weight_decay"],
    lr_scheduler_type=CONFIG["lr_scheduler_type"],
    seed=CONFIG["seed"],
    push_to_hub=CONFIG["push_to_hub"],
    output_dir=CONFIG["output_dir"],
    dataset_text_field=CONFIG["dataset_text_field"],
    packing=CONFIG["packing"],
    dataset_num_proc=CONFIG["dataset_num_proc"],
    save_steps=100,
    save_total_limit=10
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=collator,
    args=sft_config,
)

try:
    trainer.train()
except RuntimeError as e:
    print(f"Error during training: {e}")
    raise
