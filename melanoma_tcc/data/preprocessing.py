import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
import torch
from torch.utils.data import Dataset


LABEL_MAP = {0: "benign", 1: "melanoma"}


DIAGNOSIS_GROUPS = {
    "basal cell carcinoma": "BCC",
    "blue nevus": "NEV",
    "clark nevus": "NEV",
    "combined nevus": "NEV",
    "congenital nevus": "NEV",
    "dermal nevus": "NEV",
    "recurrent nevus": "NEV",
    "reed or spitz nevus": "NEV",
    "melanoma": "MEL",
    "melanoma (in situ)": "MEL",
    "melanoma (less than 0.76 mm)": "MEL",
    "melanoma (0.76 to 1.5 mm)": "MEL",
    "melanoma (more than 1.5 mm)": "MEL",
    "melanoma metastasis": "MEL",
    "seborrheic keratosis": "SK",
    "dermatofibroma": "MISC",
    "lentigo": "MISC",
    "melanosis": "MISC",
    "miscellaneous": "MISC",
    "vascular lesion": "MISC",
}

GROUP_TO_NAME = {
    "BCC": "basal cell carcinoma",
    "NEV": "nevus",
    "MEL": "melanoma",
    "SK": "seborrheic keratosis",
    "MISC": "other lesion",
}

GROUP_TO_LABEL = {"BCC": 0, "NEV": 1, "MEL": 2, "SK": 3, "MISC": 4}
LABEL_TO_GROUP = {v: k for k, v in GROUP_TO_LABEL.items()}


def build_clinical_prompt(row: pd.Series) -> str:
    sex = row.get("sex", "unknown")
    location = row.get("location", "unknown")
    elevation = row.get("elevation", "unknown")
    return (
        f"Patient: {sex}, lesion on {location}, {elevation}. "
        "Analyze the dermoscopic image and identify the most likely diagnosis "
        "among: melanoma, nevus, basal cell carcinoma, seborrheic keratosis, or other lesion. "
        "Describe the key visual features that support your assessment."
    )


def extract_visual_features(row: pd.Series) -> list:
    feature_map = [
        ("pigment_network", "pigment network"),
        ("blue_whitish_veil", "blue-whitish veil"),
        ("vascular_structures", "vascular pattern"),
        ("streaks", "streaks"),
        ("dots_and_globules", "dots and globules"),
        ("pigmentation", "pigmentation"),
        ("regression_structures", "regression structures"),
    ]
    features = []
    for col, name in feature_map:
        value = str(row.get(col, "absent")).strip().lower()
        if value and value != "absent":
            if value == "present":
                features.append(name)
            else:
                features.append(f"{value} {name}")
    return features


def build_clinical_answer(row: pd.Series) -> str:
    diagnosis_raw = str(row.get("diagnosis", "")).strip().lower()
    group = DIAGNOSIS_GROUPS.get(diagnosis_raw, "MISC")
    label_name = GROUP_TO_NAME[group]
    features = extract_visual_features(row)
    if features:
        feat_str = " and ".join(features[:2])
        return f"{feat_str.capitalize()} suggest {label_name}."
    return f"The dermoscopic features are consistent with {label_name}."


def load_image(image_path: Path, size: tuple = (224, 224)) -> Image.Image:
    return Image.open(image_path).convert("RGB").resize(size)


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    angle = rng.uniform(-10, 10)
    image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    brightness_factor = rng.uniform(0.9, 1.1)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    contrast_factor = rng.uniform(0.9, 1.1)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    return image


def balance_dataframe(df: pd.DataFrame, target_per_class: int = 80,
                      max_oversample: int = 4, seed: int = 42,
                      undersample: bool = True) -> pd.DataFrame:
    if "group" not in df.columns:
        df = df.copy()
        df["group"] = df["diagnosis"].str.strip().str.lower().map(DIAGNOSIS_GROUPS).fillna("MISC")
    balanced_parts = []
    for group_name, group_df in df.groupby("group"):
        n = len(group_df)
        if n >= target_per_class:
            if undersample:
                sampled = group_df.sample(n=target_per_class, random_state=seed, replace=False)
            else:
                sampled = group_df
        else:
            allowed_max = min(target_per_class, n * max_oversample)
            sampled = group_df.sample(n=allowed_max, random_state=seed, replace=True)
        balanced_parts.append(sampled)
    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)


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
        prompt = (
            f"Patient: {row.get('sex', 'unknown')}, "
            f"approximately {row.get('age_approx', 'unknown')} years old. "
            f"Lesion location: {row.get('anatom_site_general_challenge', 'unknown')}. "
            "Analyze the dermoscopy image of this skin lesion. "
            "Is this lesion melanoma or benign? "
            "Describe the key visual features that support your assessment."
        )
        label = int(row.get("target", -1))
        answer = LABEL_MAP.get(label, "unknown")
        return {
            "image": image,
            "prompt": prompt,
            "label": label,
            "answer": answer,
            "image_name": row["image_name"],
        }


class Derm7ptDataset(Dataset):
    def __init__(self, meta_csv: str, images_dir: str, processor,
                 indexes_csv: str = None, image_col: str = "derm",
                 balance: bool = False, target_per_class: int = 80,
                 max_oversample: int = 4, augment: bool = False, seed: int = 42):
        df = pd.read_csv(meta_csv)
        if indexes_csv is not None:
            indexes = pd.read_csv(indexes_csv)["indexes"].tolist()
            df = df.iloc[indexes].reset_index(drop=True)
        df["group"] = df["diagnosis"].str.strip().str.lower().map(DIAGNOSIS_GROUPS).fillna("MISC")
        if balance:
            df = balance_dataframe(df, target_per_class=target_per_class,
                                   max_oversample=max_oversample, seed=seed)
        self.df = df
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.image_col = image_col
        self.augment = augment
        self._aug_rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.images_dir / row[self.image_col]
        image = load_image(image_path)
        if self.augment:
            image = augment_image(image, self._aug_rng)
        prompt = build_clinical_prompt(row)
        answer = build_clinical_answer(row)
        diagnosis_raw = str(row.get("diagnosis", "")).strip().lower()
        group = DIAGNOSIS_GROUPS.get(diagnosis_raw, "MISC")
        label = GROUP_TO_LABEL[group]
        return {
            "image": image,
            "prompt": prompt,
            "label": label,
            "group": group,
            "answer": answer,
            "case_num": int(row["case_num"]),
        }


def split_dataframe(df: pd.DataFrame, val_ratio: float = 0.1, seed: int = 42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = int(len(df) * val_ratio)
    val_df = df[:val_size]
    train_df = df[val_size:]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


SEX_CATEGORIES = ["female", "male"]
LOCATION_CATEGORIES = ["abdomen", "back", "head neck", "upper limbs",
                       "lower limbs", "acral", "buttocks", "chest"]
ELEVATION_CATEGORIES = ["flat", "palpable", "nodular"]
METADATA_DIM = len(SEX_CATEGORIES) + len(LOCATION_CATEGORIES) + len(ELEVATION_CATEGORIES)


def encode_metadata(row: pd.Series) -> torch.Tensor:
    sex = str(row.get("sex", "")).strip().lower()
    location = str(row.get("location", "")).strip().lower()
    elevation = str(row.get("elevation", "")).strip().lower()
    vec = []
    for c in SEX_CATEGORIES:
        vec.append(1.0 if sex == c else 0.0)
    for c in LOCATION_CATEGORIES:
        vec.append(1.0 if location == c else 0.0)
    for c in ELEVATION_CATEGORIES:
        vec.append(1.0 if elevation == c else 0.0)
    return torch.tensor(vec, dtype=torch.float32)


class Derm7ptClassificationDataset(Dataset):
    def __init__(self, meta_csv: str, images_dir: str, processor,
                 indexes_csv: str = None, image_col: str = "derm",
                 balance: bool = False, target_per_class: int = 80,
                 max_oversample: int = 4, augment: bool = False, seed: int = 42,
                 undersample: bool = True):
        df = pd.read_csv(meta_csv)
        if indexes_csv is not None:
            indexes = pd.read_csv(indexes_csv)["indexes"].tolist()
            df = df.iloc[indexes].reset_index(drop=True)
        df["group"] = df["diagnosis"].str.strip().str.lower().map(DIAGNOSIS_GROUPS).fillna("MISC")
        if balance:
            df = balance_dataframe(df, target_per_class=target_per_class,
                                   max_oversample=max_oversample, seed=seed,
                                   undersample=undersample)
        self.df = df
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.image_col = image_col
        self.augment = augment
        self._aug_rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.images_dir / row[self.image_col]
        image = load_image(image_path, size=(224, 224))
        if self.augment:
            image = augment_image(image, self._aug_rng)
        image_processor = getattr(self.processor, "image_processor", None) or self.processor
        processed = image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        metadata = encode_metadata(row)
        diagnosis_raw = str(row.get("diagnosis", "")).strip().lower()
        group = DIAGNOSIS_GROUPS.get(diagnosis_raw, "MISC")
        label = GROUP_TO_LABEL[group]
        return {
            "pixel_values": pixel_values,
            "metadata": metadata,
            "labels": torch.tensor(label, dtype=torch.long),
            "group": group,
        }


def classification_collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "metadata": torch.stack([b["metadata"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def balanced_subset_indices(meta_csv: str, val_indexes_csv: str,
                            per_class: int = 10, seed: int = 42) -> list:
    df = pd.read_csv(meta_csv)
    val_idx = pd.read_csv(val_indexes_csv)["indexes"].tolist()
    val_df = df.iloc[val_idx].reset_index(drop=True)
    val_df["group"] = val_df["diagnosis"].str.strip().str.lower().map(DIAGNOSIS_GROUPS).fillna("MISC")
    selected = []
    for group in val_df["group"].unique():
        group_idx = val_df.index[val_df["group"] == group].tolist()
        n = min(per_class, len(group_idx))
        sample = val_df.loc[group_idx].sample(n=n, random_state=seed).index.tolist()
        selected.extend(sample)
    return [val_idx[i] for i in selected]


# ============================================================================
# v5: Schema unificado para mesclar Derm7pt + HAM10000
# ============================================================================

SEX_CATEGORIES_V5 = ["female", "male", "unknown"]
LOCATION_CATEGORIES_V5 = ["abdomen", "back", "head_neck", "upper_limbs",
                          "lower_limbs", "acral", "chest", "genital", "unknown"]
METADATA_DIM_V5 = len(SEX_CATEGORIES_V5) + len(LOCATION_CATEGORIES_V5)


DERM7PT_LOCATION_MAP = {
    "abdomen": "abdomen",
    "back": "back",
    "head neck": "head_neck",
    "upper limbs": "upper_limbs",
    "lower limbs": "lower_limbs",
    "acral": "acral",
    "buttocks": "lower_limbs",
    "chest": "chest",
    "genital areas": "genital",
}


HAM_LOCATION_MAP = {
    "abdomen": "abdomen",
    "back": "back",
    "trunk": "back",
    "chest": "chest",
    "acral": "acral",
    "face": "head_neck",
    "scalp": "head_neck",
    "ear": "head_neck",
    "neck": "head_neck",
    "upper extremity": "upper_limbs",
    "hand": "upper_limbs",
    "lower extremity": "lower_limbs",
    "foot": "lower_limbs",
    "genital": "genital",
    "unknown": "unknown",
}


HAM_DX_TO_GROUP = {
    "nv": "NEV",
    "mel": "MEL",
    "bkl": "SK",
    "bcc": "BCC",
    "akiec": "MISC",
    "vasc": "MISC",
    "df": "MISC",
}


def _onehot(value, categories):
    return [1.0 if value == c else 0.0 for c in categories]


def encode_metadata_unified(sex_raw: str, location_unified: str) -> torch.Tensor:
    sex = str(sex_raw).strip().lower()
    if sex not in SEX_CATEGORIES_V5:
        sex = "unknown"
    location = str(location_unified).strip().lower()
    if location not in LOCATION_CATEGORIES_V5:
        location = "unknown"
    vec = _onehot(sex, SEX_CATEGORIES_V5) + _onehot(location, LOCATION_CATEGORIES_V5)
    return torch.tensor(vec, dtype=torch.float32)


class Derm7ptUnifiedDataset(Dataset):
    """Versao v5 do Derm7pt usando schema unificado (sex 3 + location 9 = 12 dim)."""

    def __init__(self, meta_csv: str, images_dir: str, processor,
                 indexes_csv: str = None, image_col: str = "derm",
                 augment: bool = False, seed: int = 42):
        df = pd.read_csv(meta_csv)
        if indexes_csv is not None:
            indexes = pd.read_csv(indexes_csv)["indexes"].tolist()
            df = df.iloc[indexes].reset_index(drop=True)
        df["group"] = df["diagnosis"].str.strip().str.lower().map(DIAGNOSIS_GROUPS).fillna("MISC")
        self.df = df
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.image_col = image_col
        self.augment = augment
        self._aug_rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.images_dir / row[self.image_col]
        image = load_image(image_path, size=(224, 224))
        if self.augment:
            image = augment_image(image, self._aug_rng)
        image_processor = getattr(self.processor, "image_processor", None) or self.processor
        processed = image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        loc_raw = str(row.get("location", "")).strip().lower()
        location_unified = DERM7PT_LOCATION_MAP.get(loc_raw, "unknown")
        metadata = encode_metadata_unified(row.get("sex", ""), location_unified)
        group = row["group"]
        label = GROUP_TO_LABEL[group]
        return {
            "pixel_values": pixel_values,
            "metadata": metadata,
            "labels": torch.tensor(label, dtype=torch.long),
            "group": group,
            "source": "derm7pt",
        }


class HAM10000Dataset(Dataset):
    """HAM10000 com mesmo schema unificado do Derm7ptUnifiedDataset."""

    def __init__(self, meta_csv: str, base_dir: str, processor,
                 indexes: list = None, augment: bool = False, seed: int = 42):
        df = pd.read_csv(meta_csv)
        if indexes is not None:
            df = df.iloc[indexes].reset_index(drop=True)
        df["group"] = df["dx"].str.strip().str.lower().map(HAM_DX_TO_GROUP).fillna("MISC")
        self.df = df
        self.base_dir = Path(base_dir)
        self.processor = processor
        self.augment = augment
        self._aug_rng = random.Random(seed)

        self._image_lookup = {}
        for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            part_dir = self.base_dir / part
            if part_dir.exists():
                for f in os.listdir(part_dir):
                    if f.endswith(".jpg"):
                        self._image_lookup[f[:-4]] = part_dir / f

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        image_path = self._image_lookup.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"HAM image {image_id} not found")
        image = load_image(image_path, size=(224, 224))
        if self.augment:
            image = augment_image(image, self._aug_rng)
        image_processor = getattr(self.processor, "image_processor", None) or self.processor
        processed = image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        loc_raw = str(row.get("localization", "")).strip().lower()
        location_unified = HAM_LOCATION_MAP.get(loc_raw, "unknown")
        metadata = encode_metadata_unified(row.get("sex", ""), location_unified)
        group = row["group"]
        label = GROUP_TO_LABEL[group]
        return {
            "pixel_values": pixel_values,
            "metadata": metadata,
            "labels": torch.tensor(label, dtype=torch.long),
            "group": group,
            "source": "ham10000",
        }


class CombinedDermDataset(Dataset):
    """Concatena multiplos datasets com mesmo schema. Compativel com PyTorch DataLoader."""

    def __init__(self, datasets: list):
        self.datasets = datasets
        self.cumulative = [0]
        for d in datasets:
            self.cumulative.append(self.cumulative[-1] + len(d))

    def __len__(self):
        return self.cumulative[-1]

    def __getitem__(self, idx):
        for i in range(len(self.datasets)):
            if idx < self.cumulative[i + 1]:
                local_idx = idx - self.cumulative[i]
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range (total={self.cumulative[-1]})")


def ham10000_train_val_split(meta_csv: str, val_ratio: float = 0.15, seed: int = 42):
    df = pd.read_csv(meta_csv)
    lesions = df["lesion_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(lesions)
    n_val = int(len(lesions) * val_ratio)
    val_lesions = set(lesions[:n_val])
    val_idx = df.index[df["lesion_id"].isin(val_lesions)].tolist()
    train_idx = df.index[~df["lesion_id"].isin(val_lesions)].tolist()
    return train_idx, val_idx
