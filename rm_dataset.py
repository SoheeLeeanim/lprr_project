import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class LipPatchDataset(Dataset):
    """
    Dataset for lip patches.
    Returns a pair of (source_lip, target_lip) patches,
    applying a frame offset to the target.
    """

    def __init__(self, source_root, target_root, frame_offset_target=42):
        """
        :param source_root: Directory containing source lip patches (e.g., data/source/patch_lip).
        :param target_root: Directory containing target lip patches (e.g., data/target/patch_lip).
        :param frame_offset_target: The frame offset for the target video.
        """
        self.source_root = source_root
        self.target_root = target_root
        self.offset = frame_offset_target

       # Sort file names to ensure they are in frame order
        self.source_files = sorted(os.listdir(source_root))
        self.target_files = sorted(os.listdir(target_root))

        # --- NEW: build explicit pairs so we can reuse filenames (lip/eye) ---
        self.pairs = []

        # offset이 양수/음수 둘 다 가능하게 처리
        # target_idx = src_idx + offset
        for src_idx in range(len(self.source_files)):
            tgt_idx = src_idx + self.offset
            if 0 <= tgt_idx < len(self.target_files):
                self.pairs.append((self.source_files[src_idx], self.target_files[tgt_idx]))

        self.length = len(self.pairs)

        # Image to tensor transformation (0-255 -> 0-1 float, (H,W,C) -> (C,H,W))
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        src_name, tgt_name = self.pairs[idx]

        src_path = os.path.join(self.source_root, src_name)
        tgt_path = os.path.join(self.target_root, tgt_name)

        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        src_tensor = self.transform(src_img)
        tgt_tensor = self.transform(tgt_img)

        return src_tensor, tgt_tensor

