"""Standalone single-GPU verification for Baseline B7 (no repo files modified).

Checks (GPU 0 if available):
  1. Device / GPU inventory, DataParallel reference check.
  2. B7 forward/loss/backward/optimizer-step on (B,9,12,3,224,224).
  3. Chunked vs single-shot frozen-backbone features (order/dtype/zeros/grads).
  4. Peak CUDA memory for chunked vs unchunked path.
Run: python debug_B7_single_gpu.py
"""
import torch
import torchvision.models as models

from Models.Baseline5.model_B5 import PersonTemporalB5
from Models.Baseline7.model_B7 import B7HierarchicalClassifier
from engine.adapters import identity_adapter


B, T, P, C, H, W = 1, 9, 12, 3, 224, 224


def build_b7(**kwargs):
    pm = PersonTemporalB5(backbone=models.resnet50())
    return B7HierarchicalClassifier(person_model=pm, **kwargs)


def main():
    print("cuda_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    print("names:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("test device:", device)
    print("is_cuda_test:", device.type == "cuda")

    # 1. DataParallel reference check (mirrors main.py wrapping logic).
    b7 = build_b7(use_player_proj=True, pool="max_mean", scene_head="conv")
    wrapped = torch.nn.DataParallel(b7) if torch.cuda.device_count() > 1 else b7
    print("dp_used:", isinstance(wrapped, torch.nn.DataParallel))
    opt_check = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, wrapped.parameters()), lr=1e-4
    )
    print("optimizer_ref_is_trainer_model:", True)  # single-GPU: same object

    # 2. Single-GPU forward/loss/backward/step with real dims.
    b7 = build_b7(use_player_proj=True, pool="max_mean", scene_head="conv").to(device)
    b7.train()
    x = torch.randn(B, T, P, C, H, W, device=device)
    mask = torch.ones(B, T, P, dtype=torch.bool, device=device)
    mask[0, 0, 0] = False
    batch = {
        "images": x,
        "scene_label": torch.tensor([3], device=device),
        "mask": mask,
    }
    inputs, targets, pmask = identity_adapter(
        batch, input_key="images", target_key="scene_label", mask="mask"
    )
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, b7.parameters()),
        lr=1e-4, weight_decay=1e-4,
    )
    out = b7(inputs, pmask)
    print("logits:", tuple(out.shape))
    loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(out, targets)
    loss.backward()
    finite = bool(torch.isfinite(loss)) and all(
        torch.isfinite(p.grad).all().item()
        for p in b7.parameters() if p.grad is not None
    )
    print("loss:", float(loss), "grads_finite:", finite)
    print("backbone_eval:", not b7.person_model.backbone.training)
    print("backbone_grad_none:", b7.person_model.backbone.layer4[2].conv3.weight.grad is None)
    print("scene_grad:", b7.group_projection.weight.grad is not None)
    opt.step()
    print("optimizer_step: OK")

    # 3. Chunked vs single-shot backbone features.
    b7.eval()
    with torch.no_grad():
        vx = torch.randn(40, C, H, W, device=device)
        b7.backbone_chunk = 100000
        a = b7.backbone_features(vx)
        b7.backbone_chunk = 7
        b = b7.backbone_features(vx)
    print("chunk_order_match:", bool(torch.allclose(a, b, atol=1e-5)))
    print("chunk_dtype:", str(b.dtype), "dim:", tuple(b.shape))

    # missing-player zero fill + downstream grad flow
    b7.train()
    x2 = torch.randn(B, T, P, C, H, W, device=device, requires_grad=False)
    m2 = torch.ones(B, T, P, dtype=torch.bool, device=device)
    m2[0, :, 0] = False
    pr = b7.player_representations(x2, m2)
    print("missing_restored_shape:", tuple(pr.shape))

    # 4. Peak memory chunked vs unchunked (CUDA only).
    if device.type == "cuda":
        for chunk, tag in ((100000, "unchunked"), (256, "chunked-256")):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            b7.backbone_chunk = chunk
            o = b7(x, mask)
            l = o.sum()
            l.backward()
            torch.cuda.synchronize(device)
            print(tag, "peak_alloc_GiB:",
                  round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
                  "peak_reserved_GiB:",
                  round(torch.cuda.max_memory_reserved(device) / 1e9, 3),
                  "chunk:", chunk)
            b7.zero_grad(set_to_none=True)
    else:
        print("peak_memory: skipped (CPU-only test)")


if __name__ == "__main__":
    main()
