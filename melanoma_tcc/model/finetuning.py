import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login

MODEL_ID = "google/medgemma-4b-it"


def load_model_for_finetuning(hf_token: str):
    login(token=hf_token)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=hf_token,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    return model, processor


def apply_lora(model):
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    model.print_trainable_parameters()
    return model


class FineTuningDataset(Dataset):
    def __init__(self, base_dataset, processor):
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        tokenized = self.processor(
            text=text,
            images=sample["image"],
            return_tensors="pt",
        )
        result = {k: v.squeeze(0) for k, v in tokenized.items()}
        result["labels"] = result["input_ids"].clone()
        return result


def collate_fn(examples):
    batch = {}
    for key in examples[0].keys():
        values = [ex[key] for ex in examples]
        if key in ("input_ids", "attention_mask", "token_type_ids"):
            batch[key] = pad_sequence(values, batch_first=True, padding_value=0)
        elif key == "labels":
            batch[key] = pad_sequence(values, batch_first=True, padding_value=-100)
        elif key == "pixel_values":
            batch[key] = torch.stack(values)
        else:
            try:
                batch[key] = torch.stack(values)
            except Exception:
                batch[key] = values
    return batch


def get_trainer(model, processor, train_dataset, eval_dataset, output_dir: str):
    train_ft = FineTuningDataset(train_dataset, processor)
    eval_ft = FineTuningDataset(eval_dataset, processor)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ft,
        eval_dataset=eval_ft,
        data_collator=collate_fn,
    )
