import os
import sys

# Add project root to sys.path to allow running as script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from code.rm.models.autoencoder import RM_AutoEncoder
from tqdm import tqdm

PATCH = 128

def tensor_to_bgr(img_t):
    img = (img_t.detach().cpu().clamp(0,1).numpy().transpose(1,2,0) * 255).astype("uint8")
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def load_rgb(folder, fname):
    path = os.path.join(folder, fname)
    img = Image.open(path).convert("RGB")
    return T.ToTensor()(img)

def make_pairs(src_files, tgt_files, offset):
    pairs = []
    for s_idx, s_name in enumerate(src_files):
        t_idx = s_idx + offset
        if 0 <= t_idx < len(tgt_files):
            pairs.append((s_name, tgt_files[t_idx]))
    return pairs

def main():
    device = "cuda"
    print("[Device]", device)

    # --- File Structure Configuration ---
    data_root = "data"
    ckpt_root = "checkpoints"
    out_dir   = "samples"

    # Set checkpoint paths for each part
    ckpt_lip   = os.path.join(ckpt_root, "rm_lip",   "rm_epoch_010.pt")
    ckpt_eye_l = os.path.join(ckpt_root, "rm_eye_l", "rm_epoch_010.pt")
    ckpt_eye_r = os.path.join(ckpt_root, "rm_eye_r", "rm_epoch_010.pt")

    source_base = os.path.join(data_root, "source")
    target_base = os.path.join(data_root, "target")

    # Set the frame offset value you calculated
    frame_offset_target = -9

    parts = [
        ("lip",   "patch_lip"),
        ("eyeR",  "patch_eye_r"),
        ("eyeL",  "patch_eye_l"),
    ]

    # Sort filenames based on lip patches (assuming other parts share the same filenames)
    ref_folder = parts[0][1]
    src_lip_dir = os.path.join(source_base, ref_folder)
    tgt_lip_dir = os.path.join(target_base, ref_folder)
    src_files = sorted(os.listdir(src_lip_dir))
    tgt_files = sorted(os.listdir(tgt_lip_dir))
    pairs = make_pairs(src_files, tgt_files, frame_offset_target)
    print("[Pairs]", len(pairs))

    # Function to load the model
    def load_model(ckpt_path):
        m = RM_AutoEncoder(in_ch=3, base_ch=64, latent_ch=256).to(device)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            m.load_state_dict(ckpt["model"])
            print(f"Loaded: {ckpt_path}")
        else:
            print(f"Warning: Not found {ckpt_path}")
        m.eval()
        return m

    models = {}
    models["lip"]  = load_model(ckpt_lip)
    models["eyeL"] = load_model(ckpt_eye_l)
    models["eyeR"] = load_model(ckpt_eye_r)

    out_path = os.path.join(out_dir, "rm_preview_lip_eyeR_eyeL.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Set video dimensions: 2 rows, 6 columns (Source/Recon for Lip, EyeL, EyeR)
    H = PATCH * 2
    W = PATCH * 6
    fps = 30
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    def draw_label(img, text):
        # Draw black background rectangle for text readability
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (0, 0), (w + 4, h + 6), (0, 0, 0), -1)
        cv2.putText(img, text, (2, h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    with torch.no_grad():
        for i, (src_name, tgt_name) in enumerate(tqdm(pairs, desc="Processing Video")):
            row1_imgs = []
            row2_imgs = []

            for tag, folder in parts:
                src_dir = os.path.join(source_base, folder)
                tgt_dir = os.path.join(target_base, folder)

                src = load_rgb(src_dir, src_name).unsqueeze(0).to(device)
                tgt = load_rgb(tgt_dir, tgt_name).unsqueeze(0).to(device)

                # Use the specific model for the current part
                # src_rec: Reconstruction (Source -> Encoder -> Decoder_Source -> Source_Recon)
                # tgt_pred: Prediction (Source -> Encoder -> Decoder_Target -> Target_Prediction)
                src_rec, tgt_pred = models[tag](src)

                # Convert to BGR and add labels
                img_src  = draw_label(tensor_to_bgr(src[0]),      f"{tag} Org")
                img_rec  = draw_label(tensor_to_bgr(src_rec[0]),  f"{tag} Rec")
                img_tgt  = draw_label(tensor_to_bgr(tgt[0]),      f"{tag} Tgt")
                img_pred = draw_label(tensor_to_bgr(tgt_pred[0]), f"{tag} Pred")

                row1_imgs.extend([img_src, img_rec])
                row2_imgs.extend([img_tgt, img_pred])

            row1 = cv2.hconcat(row1_imgs)
            row2 = cv2.hconcat(row2_imgs)
            frame = cv2.vconcat([row1, row2])
            vw.write(frame)

    vw.release()
    print("saved:", out_path)

if __name__ == "__main__":
    main()
    print("Done.")
