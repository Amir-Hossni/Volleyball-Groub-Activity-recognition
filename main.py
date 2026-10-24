from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from Data.dataset import VolleyballDataset
from Data.preprocessing import prepare_model
from engine.adapters import flatten_person_batch, identity_adapter
from engine.trainer import Trainer
from Models.Baseline1.model_B1 import SceneClassifierB1
from Models.Baseline2.model_B2 import B2Model
from Models.Baseline3.model_B3 import GroupClassifierB3, PersonClassifierB3
from Models.Baseline4.model_B4 import TemporalImageClassifierB4
from Models.Baseline5.model_B5 import GroupTemporalClassifierB5, PersonTemporalB5
from Models.Baseline6.model_B6 import B6GroupActivityClassifier
from Models.Baseline7.model_B7 import B7HierarchicalClassifier
from utlis.helper_fun import (
    load_checkpoint,
    load_config,
    make_adamw,
    make_cosine_lr,
    make_reduce_lr,
    wrap_dp,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Volleyball Group-Activity Baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--baseline", "-b",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        required=True,
        help="Which baseline to train (1–8).",
    )
    parser.add_argument(
        "--stage", "-s",
        type=str,
        choices=["A", "B", "1", "2", "both"],
        default="both",
        help=(
            "For baselines with multiple stages (B3: A/B, B5: 1/2, B7: 1/2): "
            "which stage to run. 'both' runs all stages sequentially."
        ),
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working",
        help="Directory where checkpoints are saved.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        default=False,
        help="Disable Automatic Mixed Precision (AMP).",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(data_cfg: dict, args: argparse.Namespace, mode: str, image_level: bool = False):
    data_root   = Path(data_cfg["DATA_ROOT"])
    videos_path = data_root / data_cfg["PATHS"]["VIDEOS_PATH"]
    annot_root  = data_root / data_cfg["PATHS"]["TRACKING_ANNOTATION_PATH"]

    scene_to_idx  = data_cfg["CATEGORIES"]["SCENE_TO_IDX"]
    player_to_idx = data_cfg["CATEGORIES"]["PLAYER_TO_IDX"]
    train_ids     = data_cfg["SPLIT"]["TRAIN_IDS"]
    val_ids       = data_cfg["SPLIT"]["VAL_IDS"]

    transform = prepare_model(image_level=image_level)

    train_dataset = VolleyballDataset(
        videos_path=videos_path,
        annot_root=annot_root,
        split_ids=train_ids,
        scene_to_idx=scene_to_idx,
        player_to_idx=player_to_idx,
        mode=mode,
        transform=transform,
    )
    val_dataset = VolleyballDataset(
        videos_path=videos_path,
        annot_root=annot_root,
        split_ids=val_ids,
        scene_to_idx=scene_to_idx,
        player_to_idx=player_to_idx,
        mode=mode,
        transform=transform,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        drop_last=(torch.cuda.device_count() > 1),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Per-baseline training functions
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline1(args, cfg, device, scene_to_idx, player_to_idx):
    """
    Baseline 1: Single-frame scene classifier (ResNet50).
    Mode: identity, input_key='image', target_key='scene_label'
    """
    train_cfg  = cfg["Training"]
    opt_cfg    = train_cfg["OPTIMIZER"]
    out_dir    = args.output_dir
    use_amp    = not args.no_amp
    epochs     = args.epochs or 50

    train_loader, val_loader = build_loaders(
        cfg["Data"], args, mode="single_frame", image_level=True
    )

    model = SceneClassifierB1(
        num_classes=len(scene_to_idx),
        pretrained=True,
    )
    model = wrap_dp(model, device)

    optimizer = make_adamw(
        filter(lambda p: p.requires_grad, model.parameters()),
        opt_cfg,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        adapter=lambda batch: identity_adapter(
            batch, input_key="image", target_key="scene_label"
        ),
        num_classes=len(scene_to_idx),
        save_path=f"{out_dir}/best_Baseline1.pth",
        class_names=list(scene_to_idx),
        log_name="Baseline1",
        epochs=epochs,
        use_amp=use_amp,
        grad_clip=None,
    )

    print("=== Baseline 1: Training ===")
    trainer.fit(train_loader, val_loader)


def train_baseline2(args, cfg, device, scene_to_idx, player_to_idx):
    """
    Baseline 2: Multi-player scene classifier (B2Model).
    """
    train_cfg = cfg["Training"]
    opt_cfg   = train_cfg["OPTIMIZER"]
    out_dir   = args.output_dir
    use_amp   = not args.no_amp
    epochs    = args.epochs or 50

    train_loader, val_loader = build_loaders(
        cfg["Data"], args, mode="person_grouped", image_level=False
    )

    model = B2Model(
        num_players=12,
        num_classes=len(scene_to_idx),
        pretrained=True,
    )
    model = wrap_dp(model, device)

    optimizer = make_adamw(
        filter(lambda p: p.requires_grad, model.parameters()),
        opt_cfg,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        adapter=lambda batch: identity_adapter(
            batch, input_key="images", target_key="scene_label"
        ),
        num_classes=len(scene_to_idx),
        save_path=f"{out_dir}/best_Baseline2.pth",
        class_names=list(scene_to_idx),
        log_name="Baseline2",
        epochs=epochs,
        use_amp=use_amp,
        grad_clip=None,
    )

    print("=== Baseline 2: Training ===")
    trainer.fit(train_loader, val_loader)


def train_baseline3(args, cfg, device, scene_to_idx, player_to_idx):
    """
    Baseline 3: Two-stage person → group classifier.
      Stage A: Train PersonClassifierB3 on player actions.
      Stage B: Freeze backbone, train GroupClassifierB3 on scene labels.
    """
    train_cfg = cfg["Training"]
    b3a_cfg   = train_cfg["B3_STAGE_A"]["OPTIMIZER"]
    opt_cfg   = train_cfg["OPTIMIZER"]
    sch_cfg   = train_cfg["SCHEDULER"]
    out_dir   = args.output_dir
    use_amp   = not args.no_amp

    run_stage_a = args.stage in ("A", "1", "both")
    run_stage_b = args.stage in ("B", "2", "both")

    # ── Stage A ──────────────────────────────────────────────────────────────
    if run_stage_a:
        train_loader_a, val_loader_a = build_loaders(
            cfg["Data"], args, mode="person_frames", image_level=False
        )

        person_model = PersonClassifierB3(
            num_classes=len(player_to_idx),
            pretrained=True,
        )
        person_model = wrap_dp(person_model, device)

        optimizer_a  = make_adamw(person_model.parameters(), b3a_cfg)
        criterion_a  = nn.CrossEntropyLoss()

        trainer_a = Trainer(
            model=person_model,
            optimizer=optimizer_a,
            criterion=criterion_a,
            device=device,
            adapter=lambda batch: flatten_person_batch(
                batch, input_key="images", target_key="player_labels",
                ignore_index=-1,
            ),
            num_classes=len(player_to_idx),
            save_path=f"{out_dir}/best_B3_person_stage1.pth",
            class_names=list(player_to_idx),
            log_name="B3_StageA",
            epochs=args.epochs or 5,
            use_amp=use_amp,
            grad_clip=None,
        )
        print("=== Baseline 3 – Stage A: Person Classifier ===")
        trainer_a.fit(train_loader_a, val_loader_a)

    # ── Stage B ──────────────────────────────────────────────────────────────
    if run_stage_b:
        train_loader_b, val_loader_b = build_loaders(
            cfg["Data"], args, mode="person_grouped", image_level=False
        )

        ckpt_path = f"{out_dir}/best_B3_person_stage1.pth"
        bare_person = PersonClassifierB3(
            num_classes=len(player_to_idx),
            pretrained=False,
        )
        load_checkpoint(bare_person, ckpt_path, device)

        backbone = bare_person.model
        backbone.fc = nn.Identity()

        group_model = GroupClassifierB3(
            backbone=backbone,
            num_players=12,
            num_classes=len(scene_to_idx),
        )
        group_model = wrap_dp(group_model, device)

        optimizer_b = make_adamw(
            filter(lambda p: p.requires_grad, group_model.parameters()),
            opt_cfg,
        )
        scheduler_b = make_reduce_lr(optimizer_b, sch_cfg)
        criterion_b = nn.CrossEntropyLoss(label_smoothing=0.1)

        trainer_b = Trainer(
            model=group_model,
            optimizer=optimizer_b,
            criterion=criterion_b,
            device=device,
            adapter=lambda batch: identity_adapter(
                batch, input_key="images", target_key="scene_label"
            ),
            num_classes=len(scene_to_idx),
            save_path=f"{out_dir}/best_B3_group_stage2.pth",
            class_names=list(scene_to_idx),
            log_name="B3_StageB",
            epochs=args.epochs or 50,
            use_amp=use_amp,
            grad_clip=None,
            scheduler=scheduler_b,
        )
        print("=== Baseline 3 – Stage B: Group Classifier ===")
        trainer_b.fit(train_loader_b, val_loader_b)


def train_baseline4(args, cfg, device, scene_to_idx, player_to_idx):
    """
    Baseline 4: Temporal image classifier (ResNet50 + LSTM over frames).
    Requires best_Baseline1.pth for the backbone.
    """
    train_cfg = cfg["Training"]
    opt_cfg   = train_cfg["OPTIMIZER"]
    out_dir   = args.output_dir
    use_amp   = not args.no_amp
    epochs    = args.epochs or 50

    train_loader, val_loader = build_loaders(
        cfg["Data"], args, mode="clip_frames", image_level=True
    )

    b1 = SceneClassifierB1(
        num_classes=len(scene_to_idx),
        pretrained=False,
    )
    load_checkpoint(b1, f"{out_dir}/best_Baseline1.pth", device)

    backbone = b1.model
    model = TemporalImageClassifierB4(
        num_classes=len(scene_to_idx),
        backbone=backbone,
    )
    model = wrap_dp(model, device)

    optimizer = make_adamw(
        filter(lambda p: p.requires_grad, model.parameters()),
        opt_cfg,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        adapter=lambda batch: identity_adapter(
            batch, input_key="frames", target_key="scene_label"
        ),
        num_classes=len(scene_to_idx),
        save_path=f"{out_dir}/best_Baseline4.pth",
        class_names=list(scene_to_idx),
        log_name="Baseline4",
        epochs=epochs,
        use_amp=use_amp,
        grad_clip=None,
    )

    print("=== Baseline 4: Temporal Image Classifier ===")
    trainer.fit(train_loader, val_loader)


def train_baseline5(args, cfg, device, scene_to_idx, player_to_idx):
    """
    Baseline 5: Person-temporal (LSTM1 per player) + Group LSTM.
      Stage 1: PersonTemporalB5 on player actions.
      Stage 2: GroupTemporalClassifierB5 on scene labels.
    Requires best_B3_person_stage1.pth for the shared backbone.
    """
    train_cfg = cfg["Training"]
    opt_cfg   = train_cfg["OPTIMIZER"]
    sch_cfg   = train_cfg["SCHEDULER"]
    out_dir   = args.output_dir
    use_amp   = not args.no_amp

    run_s1 = args.stage in ("1", "A", "both")
    run_s2 = args.stage in ("2", "B", "both")

    train_loader, val_loader = build_loaders(
        cfg["Data"], args, mode="person_temporal", image_level=False
    )

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if run_s1:
        bare_b3 = PersonClassifierB3(
            num_classes=len(player_to_idx),
            pretrained=False,
        )
        load_checkpoint(bare_b3, f"{out_dir}/best_B3_person_stage1.pth", device)
        backbone_b5 = bare_b3.model
        backbone_b5.fc = nn.Identity()

        model_s1 = PersonTemporalB5(
            backbone=backbone_b5,
            num_classes=len(player_to_idx),
            lstm_hidden=512,
            lstm_layers=1,
            dropout=0.2,
        )
        model_s1 = wrap_dp(model_s1, device)

        optimizer_s1 = make_adamw(
            filter(lambda p: p.requires_grad, model_s1.parameters()),
            opt_cfg,
        )
        scheduler_s1 = make_reduce_lr(optimizer_s1, sch_cfg)
        criterion_s1 = nn.CrossEntropyLoss(label_smoothing=0.1)

        trainer_s1 = Trainer(
            model=model_s1,
            optimizer=optimizer_s1,
            criterion=criterion_s1,
            device=device,
            adapter=lambda batch: flatten_person_batch(
                batch, input_key="images", target_key="player_labels",
                ignore_index=-1,
            ),
            num_classes=len(player_to_idx),
            save_path=f"{out_dir}/best_Baseline5_stage1.pth",
            class_names=list(player_to_idx),
            log_name="Baseline5_stage1",
            epochs=args.epochs or 50,
            use_amp=use_amp,
            grad_clip=1.0,
            scheduler=scheduler_s1,
        )
        print("=== Baseline 5 – Stage 1: Person Temporal ===")
        trainer_s1.fit(train_loader, val_loader)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if run_s2:
        bare_b3 = PersonClassifierB3(
            num_classes=len(player_to_idx),
            pretrained=False,
        )
        load_checkpoint(bare_b3, f"{out_dir}/best_B3_person_stage1.pth", device)
        backbone_b5s2 = bare_b3.model
        backbone_b5s2.fc = nn.Identity()

        person_model_s2 = PersonTemporalB5(
            backbone=backbone_b5s2,
            num_classes=len(player_to_idx),
            lstm_hidden=512,
            lstm_layers=1,
            dropout=0.2,
        )
        load_checkpoint(person_model_s2, f"{out_dir}/best_Baseline5_stage1.pth", device)

        model_s2 = GroupTemporalClassifierB5(person_model=person_model_s2)
        model_s2 = wrap_dp(model_s2, device)

        optimizer_s2 = make_adamw(
            filter(lambda p: p.requires_grad, model_s2.parameters()),
            opt_cfg,
        )
        scheduler_s2 = make_reduce_lr(optimizer_s2, sch_cfg)
        criterion_s2 = nn.CrossEntropyLoss(label_smoothing=0.1)

        trainer_s2 = Trainer(
            model=model_s2,
            optimizer=optimizer_s2,
            criterion=criterion_s2,
            device=device,
            adapter=lambda batch: identity_adapter(
                batch, input_key="images", target_key="scene_label",
                mask="player_mask",
            ),
            num_classes=len(scene_to_idx),
            save_path=f"{out_dir}/best_Baseline5_stage2.pth",
            class_names=list(scene_to_idx),
            log_name="Baseline5_stage2",
            epochs=args.epochs or 50,
            use_amp=use_amp,
            grad_clip=1.0,
            scheduler=scheduler_s2,
        )
        print("=== Baseline 5 – Stage 2: Group Temporal ===")
        trainer_s2.fit(train_loader, val_loader)


def train_baseline6(args, cfg, device, scene_to_idx, player_to_idx):
    """
    Baseline 6: Group activity classifier (B6 architecture).
    Requires best_B3_person_stage1.pth for the backbone.
    """
    train_cfg = cfg["Training"]
    opt_cfg   = train_cfg["OPTIMIZER"]
    sch_cfg   = train_cfg["SCHEDULER"]
    out_dir   = args.output_dir
    use_amp   = not args.no_amp
    epochs    = args.epochs or 50

    train_loader, val_loader = build_loaders(
        cfg["Data"], args, mode="clip_frames_players", image_level=False
    )

    bare_b3 = PersonClassifierB3(
        num_classes=len(player_to_idx),
        pretrained=False,
    )
    load_checkpoint(bare_b3, f"{out_dir}/best_B3_person_stage1.pth", device)
    backbone_b6 = bare_b3.model
    backbone_b6.fc = nn.Identity()

    model = B6GroupActivityClassifier(backbone=backbone_b6)
    model = wrap_dp(model, device)

    optimizer = make_adamw(
        filter(lambda p: p.requires_grad, model.parameters()),
        opt_cfg,
    )
    scheduler = make_reduce_lr(optimizer, sch_cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        adapter=lambda batch: identity_adapter(
            batch, input_key="images", target_key="scene_label", mask="mask"
        ),
        num_classes=len(scene_to_idx),
        save_path=f"{out_dir}/best_Baseline6.pth",
        class_names=list(scene_to_idx),
        log_name="Baseline6",
        epochs=epochs,
        use_amp=use_amp,
        grad_clip=None,
        scheduler=scheduler,
    )

    print("=== Baseline 6: Group Activity Classifier ===")
    trainer.fit(train_loader, val_loader)


def train_baseline7(args, cfg, device, scene_to_idx, player_to_idx):
    """
    Baseline 7: Hierarchical classifier (B3 backbone + LSTM1 + LSTM2).
      Phase 1 – Linear Probe: only group-level layers train (person frozen).
      Phase 2 – Joint Fine-tune: LSTM1 + player_proj open at low LR;
                                  group layers at higher LR; CosineAnnealingLR.

    Requires best_B3_person_stage1.pth and best_Baseline5_stage1.pth.
    """
    train_cfg   = cfg["Training"]
    b7p_cfg     = train_cfg["B7_PROBE"]["OPTIMIZER"]
    b7f_cfg     = train_cfg["B7_FINETUNE"]
    b7f_sch_cfg = b7f_cfg["SCHEDULER"]
    out_dir     = args.output_dir
    use_amp     = not args.no_amp

    run_probe    = args.stage in ("1", "A", "both")
    run_finetune = args.stage in ("2", "B", "both")

    train_loader, val_loader = build_loaders(
        cfg["Data"], args, mode="clip_frames_players", image_level=False
    )

    # ── Build person model (PersonTemporalB5) ────────────────────────────────
    bare_b3 = PersonClassifierB3(
        num_classes=len(player_to_idx),
        pretrained=False,
    )
    load_checkpoint(bare_b3, f"{out_dir}/best_B3_person_stage1.pth", device)
    backbone_b7 = bare_b3.model
    backbone_b7.fc = nn.Identity()

    person_model = PersonTemporalB5(
        backbone=backbone_b7,
        num_classes=len(player_to_idx),
        lstm_hidden=512,
        lstm_layers=1,
        dropout=0.2,
    )
    load_checkpoint(person_model, f"{out_dir}/best_Baseline5_stage1.pth", device)

    # ── Build B7 (unwrapped) ─────────────────────────────────────────────────
    model_B7 = B7HierarchicalClassifier(
        person_model=person_model,
        num_classes=len(scene_to_idx),
        group_feature_dim=3000,
        hidden_dim=500,
        feature_dropout=0.0,
        use_player_proj=False,
        pool="max_mean",
        scene_head="conv",
    )

    # ── DataParallel wrapper (trainers always receive this) ───────────────────
    wrapped_model_B7 = wrap_dp(model_B7, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

    def b7_adapter(batch):
        return identity_adapter(
            batch, input_key="images", target_key="scene_label", mask="mask"
        )

    probe_best_acc = 0.0

    # ── Phase 1: Linear Probe ────────────────────────────────────────────────
    if run_probe:
        probe_params = [
            {"params": model_B7.group_norm.parameters()},
            {"params": model_B7.group_projection.parameters()},
            {"params": model_B7.lstm.parameters()},
            {"params": model_B7.classifier.parameters()},
        ]
        if model_B7.scene_skip_proj is not None:
            probe_params.extend([
                {"params": model_B7.scene_skip_proj.parameters()},
                {"params": model_B7.scene_conv1.parameters()},
                {"params": model_B7.scene_conv2.parameters()},
            ])

        optimizer_probe = torch.optim.AdamW(
            probe_params,
            lr=b7p_cfg["lr"],
            weight_decay=b7p_cfg["weight_decay"],
        )

        probe_epochs = 10 if args.epochs is None else args.epochs

        trainer_probe = Trainer(
            model=wrapped_model_B7,
            optimizer=optimizer_probe,
            criterion=criterion,
            device=device,
            adapter=b7_adapter,
            num_classes=len(scene_to_idx),
            save_path=f"{out_dir}/best_Baseline7.pth",
            class_names=list(scene_to_idx),
            log_name="Baseline7_probe",
            epochs=probe_epochs,
            early_stop_patience=10,
            use_amp=use_amp,
            grad_clip=1.0,
            scheduler=None,
        )

        print("=== Baseline 7 – Phase 1: Linear Probe ===")
        trainer_probe.fit(train_loader, val_loader)
        probe_best_acc = trainer_probe.best_acc

    # ── Phase 2: Joint Fine-tune ─────────────────────────────────────────────
    if run_finetune:
        actual_model = (
            wrapped_model_B7.module
            if hasattr(wrapped_model_B7, "module")
            else wrapped_model_B7
        )

        # Reload best Phase 1 probe checkpoint if available so Phase 2 starts from the best state
        probe_ckpt_path = Path(f"{out_dir}/best_Baseline7.pth")
        if probe_ckpt_path.exists():
            load_checkpoint(actual_model, str(probe_ckpt_path), device)

        lstm1_params = actual_model.unfreeze_person_lstm()
        print("LSTM1 unfrozen for Phase 2")

        lr_person = b7f_cfg["LR_PERSON"]
        wd_person = b7f_cfg["WEIGHT_DECAY_PERSON"]
        lr_group  = b7f_cfg["LR_GROUP"]
        wd_group  = b7f_cfg["WEIGHT_DECAY_GROUP"]

        finetune_params = [
            # LSTM1: low lr to preserve Stage-A person representations
            {"params": lstm1_params, "lr": lr_person, "weight_decay": wd_person},
            # group-level components: higher lr
            {"params": actual_model.group_norm.parameters(), "lr": lr_group, "weight_decay": wd_group},
            {"params": actual_model.group_projection.parameters(), "lr": lr_group, "weight_decay": wd_group},
            {"params": actual_model.lstm.parameters(), "lr": lr_group, "weight_decay": wd_group},
            {"params": actual_model.classifier.parameters(), "lr": lr_group, "weight_decay": wd_group},
        ]
        if actual_model.player_proj is not None:
            finetune_params.append(
                {"params": actual_model.player_proj.parameters(), "lr": lr_person, "weight_decay": wd_person}
            )
        if actual_model.scene_skip_proj is not None:
            finetune_params.extend([
                {"params": actual_model.scene_skip_proj.parameters(), "lr": lr_group, "weight_decay": wd_group},
                {"params": actual_model.scene_conv1.parameters(), "lr": lr_group, "weight_decay": wd_group},
                {"params": actual_model.scene_conv2.parameters(), "lr": lr_group, "weight_decay": wd_group},
            ])

        optimizer_finetune = torch.optim.AdamW(finetune_params)

        finetune_epochs = args.epochs or 50

        scheduler_finetune = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_finetune,
            T_max=finetune_epochs,
            eta_min=b7f_sch_cfg["eta_min"],
        )

        trainer_finetune = Trainer(
            model=wrapped_model_B7,
            optimizer=optimizer_finetune,
            criterion=criterion,
            device=device,
            adapter=b7_adapter,
            num_classes=len(scene_to_idx),
            save_path=f"{out_dir}/best_Baseline7.pth",
            class_names=list(scene_to_idx),
            log_name="Baseline7_finetune",
            epochs=finetune_epochs,
            early_stop_patience=15,
            use_amp=use_amp,
            grad_clip=1.0,
            scheduler=scheduler_finetune,
        )
        # Retain probe's best accuracy threshold so Phase 2 only saves when it outperforms Phase 1
        if probe_best_acc > 0:
            trainer_finetune.best_acc = probe_best_acc

        print("=== Baseline 7 – Phase 2: Joint Fine-tune ===")
        trainer_finetune.fit(train_loader, val_loader)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_FNS = {
    1: train_baseline1,
    2: train_baseline2,
    3: train_baseline3,
    4: train_baseline4,
    5: train_baseline5,
    6: train_baseline6,
    7: train_baseline7,
}


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"[device] {device}  |  GPUs available: {num_gpus}")

    data_cfg = cfg["Data"]
    scene_to_idx  = data_cfg["CATEGORIES"]["SCENE_TO_IDX"]
    player_to_idx = data_cfg["CATEGORIES"]["PLAYER_TO_IDX"]

    # ── Output dir ───────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    train_fn = BASELINE_FNS[args.baseline]
    train_fn(args, cfg, device, scene_to_idx, player_to_idx)


if __name__ == "__main__":
    main()
