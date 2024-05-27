import sys
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def main(hf_token, csv_path, model_name, max_steps, push_entire_model, push_lora_adapters):
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["Instruction"]
        inputs = examples["Prompt"]
        outputs = examples["Response"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    log_dir = f"logs/{model_name}-log"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=int(max_steps),
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            logging_dir=log_dir,  # Enable TensorBoard logging
            report_to="tensorboard",
        ),
    )

    trainer_stats = trainer.train()

    # Handling push options
    if push_entire_model.lower() == "true":
        model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
        model.push_to_hub_merged(model_name, tokenizer, save_method="merged_16bit", token=hf_token)
    
    if push_lora_adapters.lower() == "true":
        model.save_pretrained_merged("model", tokenizer, save_method="lora")
        model.push_to_hub_merged(f"{model_name}-lora", tokenizer, save_method="lora", token=hf_token)

if __name__ == "__main__":
    hf_token = sys.argv[1]
    csv_path = sys.argv[2]
    model_name = sys.argv[3]
    max_steps = sys.argv[4]
    push_entire_model = sys.argv[5]  # Added argument for push entire model option
    push_lora_adapters = sys.argv[6]  # Added argument for push LoRA adapters option

    main(hf_token, csv_path, model_name, max_steps, push_entire_model, push_lora_adapters)
