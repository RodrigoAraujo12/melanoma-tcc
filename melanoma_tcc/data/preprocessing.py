import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset


LABEL_MAP = {0: "benign", 1: "melanoma"}


def build_clinical_prompt(row: pd.Series) -> str:
    age = row.get("age_approx", "unknown")
    sex = row.get("sex", "unknown")
    site = row.get("anatom_site_general_challenge", "unknown")
    return (
        f"Patient: {sex}, approximately {age} years old. "
        f"Lesion location: {site}. "
        "Analyze the dermoscopy image of this skin lesion. "
        "Is this lesion melanoma or benign? "
        "Describe the key visual features that support your assessment."
    )


def load_image(image_path: Path, size: tuple = (224, 224)) -> Image.Image:
    return Image.open(image_path).convert("RGB").resize(size)


class ISICDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, processor, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.images_dir / f"{row['image_name']}.jpg"
        image = load_image(image_path)
        prompt = build_clinical_prompt(row)
        label = int(row.get("target", -1))
        answer = LABEL_MAP.get(label, "unknown")
        return {
            "image": image,
            "prompt": prompt,
            "label": label,
            "answer": answer,
            "image_name": row["image_name"],
        }


def split_dataframe(df: pd.DataFrame, val_ratio: float = 0.1, seed: int = 42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = int(len(df) * val_ratio)
    val_df = df[:val_size]
    train_df = df[val_size:]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
