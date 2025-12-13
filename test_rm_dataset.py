from rm_dataset import LipPatchDataset

if __name__ == "__main__":
    # Example: source frame 26 ≈ target frame 68 → offset ≈ 42
    dataset = LipPatchDataset(
        source_root="data/source/patch_lip",
        target_root="data/target/patch_lip",
        frame_offset_target=42,   # This value can be adjusted
    )

    print("Total number of samples:", len(dataset))

    src, tgt = dataset[0]
    print("source[0] shape:", src.shape)
    print("target[0+42] shape:", tgt.shape)

    # For debugging: to check which files are actually being matched
    # print(dataset.source_files[0], dataset.target_files[0 + 42])
