import argparse
import os
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_checkpoint", type=str, required=True)
    p.add_argument(
        "--gates_path",
        type=str,
        required=True,
        help="gates map (.pt or .json), {layer_name: [bits,...]}",
    )
    p.add_argument("--task_suite_name", type=str, default="libero_spatial")
    p.add_argument("--num_trials_per_task", type=int, default=1)
    p.add_argument("--local_log_dir", type=str, default="rollouts_qvla_fakew")
    p.add_argument("--libero_root", type=str, default=os.environ.get("LIBERO_ROOT", ""))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    openvla_src_root = repo_root
    if openvla_src_root not in sys.path:
        sys.path.insert(0, openvla_src_root)
    if args.libero_root:
        if os.path.isdir(args.libero_root) and args.libero_root not in sys.path:
            sys.path.insert(0, args.libero_root)

    from qvla.inject_fake_w import inject_qvla_weight_fake_quant
    import experiments.robot.libero.run_libero_eval as L
    from experiments.robot.libero.run_libero_eval import eval_libero
    import experiments.robot.robot_utils as R
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_checkpoint,
        attn_implementation=("flash_attention_2" if device.type == "cuda" else "eager"),
        torch_dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    dataset_statistics_path = os.path.join(args.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        import json

        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        model.norm_stats = norm_stats

    injected = inject_qvla_weight_fake_quant(
        model,
        gates_path=args.gates_path,
        device=device,
    )
    print(f"[qvla][fake-w] injected into {injected} modules")

    _orig_get_model = R.get_model
    _orig_L_get_model = getattr(L, "get_model", None)

    def _patched_get_model(cfg_in):
        print("[qvla][fake-w] using preloaded model")
        return model

    R.get_model = _patched_get_model  # type: ignore
    if _orig_L_get_model is not None:
        L.get_model = _patched_get_model  # type: ignore

    try:
        _argv_bak = list(sys.argv)
        sys.argv = [
            sys.argv[0],
            "--model_family",
            "openvla",
            "--pretrained_checkpoint",
            str(args.pretrained_checkpoint),
            "--task_suite_name",
            str(args.task_suite_name),
            "--num_trials_per_task",
            str(args.num_trials_per_task),
            "--local_log_dir",
            str(args.local_log_dir),
            "--center_crop",
            str(True),
            "--seed",
            str(args.seed),
        ]
        os.makedirs(args.local_log_dir, exist_ok=True)
        eval_libero()
    finally:
        R.get_model = _orig_get_model  # type: ignore
        if _orig_L_get_model is not None:
            L.get_model = _orig_L_get_model  # type: ignore
        sys.argv = _argv_bak


if __name__ == "__main__":
    main()
