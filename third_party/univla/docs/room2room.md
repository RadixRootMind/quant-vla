## R2R Deployment
### Setup
we focus on the Room2Room (R2R) task in VLN-CE, one of the most widely recognized benchmarks in vision-and-language navigation (VLN).
VLN-CE uses [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) 0.1.7 which can be [built from source](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7#installation) or installed from conda:

```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
```
Then install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7):

```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

Now you can install VLN-CE:

```bash
git clone git@github.com:jacobkrantz/VLN-CE.git
cd VLN-CE
python -m pip install -r requirements.txt
```

### Data

#### Scenes: Matterport3D

Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script (`download_mp.py`) can be accessed by following the instructions on their [project webpage](https://niessner.github.io/Matterport/). The scene data can then be downloaded:

```bash
python download_mp.py --task habitat -o datasets/scene_datasets/mp3d/
```

Extract so that it takes the form of `datasets/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes.

#### MetaData

The R2R_VLNCE dataset is a port of the Room-to-Room (R2R) dataset created by [Anderson et al](http://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf) for use with the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) (MP3D-Sim). For details on porting to 3D reconstructions, please see our [paper](https://arxiv.org/abs/2004.02857). `R2R_VLNCE_v1-3` is a minimal version of the dataset. See the [dataset page](https://jacobkrantz.github.io/vlnce/data) for format, contents, and a changelog. We encourage use of the most recent version (`v1-3`).

Downloading via CLI:

```bash
# R2R_VLNCE_v1-3
gdown https://drive.google.com/uc?id=1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm
```

#### Data Collection

Considering that R2R dataset does not contain RGB image observations at each moment within the trajectory, it is necessary to rollout the reference path in the VLNCE environment to collect image observations for following training. We provide the corresponding code. First, replace the file `VLN-CE/vlnce_baselines/config/r2r_baselines/nonlearning.yaml` with `experiments/robot/r2r/config/nonlearning.yaml`. Then, replace `VLN-CE/vlnce_baselines/nonlearning_agents.py` with `experiments/robot/r2r/nonlearning_agents.py`. Next, run the following command to collect data:

```bash
cd VLN-CE/
python run.py --exp-config vlnce_baselines/config/r2r_baselines/nonlearning.yaml --run-type eval
```

### Training

```bash
cd ./vla-scripts
torchrun --standalone --nnodes 1 --nproc-per-node 8 finetune_r2r.py \
--vla_path <vla_path> \
--lam_path <lam_path> \
--data_root_dir <data_root_dir> \
--run_root_dir <run_root_dir>
```

### Evaluation

We evaluate on the 1,839 samples in the R2R val-unseen split

```bash
python experiments/robot/r2r/run_r2r_eval.py \
--action_decoder_path   <action_decoder_path> \
--pretrained_checkpoint <pretrained_checkpoint> 
```


