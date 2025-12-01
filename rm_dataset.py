import os                                # 운영체제 기능 (경로, 파일 리스트 등)
from PIL import Image                    # 이미지 열기 (Python Imaging Library)
from torch.utils.data import Dataset     # PyTorch Dataset 기본 클래스
import torchvision.transforms as T       # transforms = 이미지 전처리 도구 모음


class LipPatchDataset(Dataset):
    """
    입술 패치용 Dataset
    source_lip 패치와 target_lip 패치를 '프레임 오프셋'을 적용해서 쌍으로 반환
    """

    def __init__(self, source_root, target_root, frame_offset_target=42):
        """
        source_root: source 입술 패치 폴더 (예: data/source/patch_lip)
        target_root: target 입술 패치 폴더 (예: data/target/patch_lip)
        frame_offset_target: 타겟 프레임이 얼마나 뒤에 있는지 (정수)
        """
        self.source_root = source_root
        self.target_root = target_root
        self.offset = frame_offset_target   # 몇 프레임 밀려 있는지 저장

        # 폴더 안의 파일 이름을 정렬해서 '프레임 순서'로 사용
        self.source_files = sorted(os.listdir(source_root))
        self.target_files = sorted(os.listdir(target_root))

        # 쌍으로 만들 수 있는 최대 개수
        # target 쪽은 offset 만큼 앞부분을 버린다고 생각하면 됨
        self.length = min(len(self.source_files),
                          len(self.target_files) - self.offset)

        # 이미지 -> 텐서 변환 (0~255 -> 0~1 float, (H,W,C) -> (C,H,W))
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        """데이터셋 크기(사용 가능한 source-target 쌍의 개수)"""
        return self.length

    def __getitem__(self, idx):
        """
        idx번째 쌍을 반환:
        - source: source_files[idx]
        - target: target_files[idx + offset]
        """
        src_name = self.source_files[idx]
        tgt_name = self.target_files[idx + self.offset]

        src_path = os.path.join(self.source_root, src_name)
        tgt_path = os.path.join(self.target_root, tgt_name)

        # 이미지 열기 (RGB 강제)
        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        # 텐서 변환
        src_tensor = self.transform(src_img)   # (3, 128, 128)
        tgt_tensor = self.transform(tgt_img)   # (3, 128, 128)

        return src_tensor, tgt_tensor
