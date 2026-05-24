import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
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
