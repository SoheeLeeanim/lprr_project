from rm_dataset import LipPatchDataset

if __name__ == "__main__":
    # source 26프레임 ≈ target 68프레임 → offset ≈ 42
    dataset = LipPatchDataset(
        source_root="data/source/patch_lip",
        target_root="data/target/patch_lip",
        frame_offset_target=42,   # 여기 숫자만 바꿔가며 조절 가능
    )

    print("총 샘플 수:", len(dataset))

    src, tgt = dataset[0]
    print("source[0] shape:", src.shape)
    print("target[0+42] shape:", tgt.shape)

    # 확인용: 실제 어느 파일이 매칭되는지도 보고 싶으면
    # (디버깅용 코드 — 필요하면 써보기)
