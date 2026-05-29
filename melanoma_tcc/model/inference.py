import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login

MODEL_ID = "google/medgemma-4b-it"


def load_model(hf_token: str, device: str = "auto"):
    login(token=hf_token)
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, processor


def load_finetuned_model(hf_token: str, adapter_path: str, device: str = "auto"):
    login(token=hf_token)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=hf_token,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, processor


def predict(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[-1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


def extract_label_from_response(response: str) -> int:
    response_lower = response.lower()
    if "melanoma" in response_lower and "benign" not in response_lower:
        return 1
    if "benign" in response_lower:
        return 0
    return -1


import re


CLASS_KEYWORDS = {
    "BCC": ["basal cell carcinoma", "basal-cell carcinoma", "bcc"],
    "SK": ["seborrheic keratosis", "seborrhoeic keratosis"],
    "MEL": ["melanoma"],
    "NEV": ["nevus", "nevi", "naevus"],
    "MISC": ["dermatofibroma", "lentigo", "melanosis", "vascular lesion", "other lesion", "miscellaneous"],
}

GROUP_TO_LABEL = {"BCC": 0, "NEV": 1, "MEL": 2, "SK": 3, "MISC": 4}


def _match_group_in_text(text: str) -> str:
    text = text.lower()
    for group, keywords in CLASS_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return group
    return None


def extract_multiclass_label(response: str) -> int:
    bold_matches = re.findall(r"\*\*([^*]+)\*\*", response)
    for match in bold_matches:
        group = _match_group_in_text(match)
        if group is not None:
            return GROUP_TO_LABEL[group]

    patterns = [
        r"diagnosis is[:\s]+([^.\n]+)",
        r"most likely diagnosis is[:\s]+([^.\n]+)",
        r"consistent with[:\s]+([^.\n]+)",
        r"suggest(?:s|ive of)?[:\s]+([^.\n]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, response.lower())
        if m:
            group = _match_group_in_text(m.group(1))
            if group is not None:
                return GROUP_TO_LABEL[group]

    group = _match_group_in_text(response)
    if group is not None:
        return GROUP_TO_LABEL[group]
    return -1
