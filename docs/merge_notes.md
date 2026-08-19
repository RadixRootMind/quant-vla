# Merge Notes

This freeze records the current code-level integration state before the next UniVLA merge phase.

## Merged Project Families

| Source family | Current status | Main files |
| --- | --- | --- |
| QuantVLA | Integrated | `gr00t/quantization/`, `scripts/run_groot_benchmark.sh`, `scripts/run_awesome_quant_vla.sh` |
| Omega-QVLA | Integrated | GPTQ builders, Pi0.5 runtime/eval scripts, W4A4 pack route |
| QVLA/OpenVLA | Integrated | `third_party/openvla`, `third_party/openvla_oft`, `tools/qvla`, `scripts/run_openvla_qvla.sh` |
| UniVLA | Not merged yet | Local source identified as `D:\VLA\QVLA\UniVLA` / OpenDriveLab/UniVLA |

## Current Validation Baseline

- GR00T-N1.5 W4A8 on LIBERO Object: 82.0%.
- Pi0.5 W4A8 on LIBERO Object: 99.0%.
- Pi0.5 W4A4 GPTQ on LIBERO Object: 98.0%.
- OpenVLA FP16 on LIBERO Spatial: 70.0%.
- OpenVLA QVLA W8 on LIBERO Spatial: 80.0%.
- OpenVLA-OFT FP16 on LIBERO Spatial: 60.0%.
- OpenVLA-OFT QVLA W8 on LIBERO Spatial: 26.0%.

## UniVLA Identification

The local `D:\VLA\QVLA\UniVLA` directory corresponds to OpenDriveLab/UniVLA:

- README title: `UniVLA`.
- Project site: `opendrivelab.com`.
- Paper: `https://arxiv.org/pdf/2505.06111`.
- Upstream clone command in README: `git clone git@github.com:OpenDriveLab/UniVLA.git`.
- Hugging Face models in README use the `qwbu/univla-*` namespace.

It is a generalist VLA project with a task-centric latent action model. It should be merged as a fourth backend family after this freeze, not folded into the existing OpenVLA/QVLA route blindly.

