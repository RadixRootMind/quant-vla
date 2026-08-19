## Real-World Deployment

### Finetune

#### Data Pre-processing


> [!TIP]
>  If your datasets are in LeRobot format, you can convert them using the LeRobot2RLDS tool from [Any4Lerobot](https://github.com/Tavish9/any4lerobot). After conversion, you can load the data using the same pipeline we employed for LIBERO training.

- Here we provide a script to process hdf5 files from an Agilex arm: ```/prismatic/vla/datasets/real_world_dataset.py```
- The data is structured as follows:

```
├── action
├── observations
│   ├── images
│   │   ├── cam_high
│   ├── qpos
```

The descriptions of each key above are as follows:

| Keys     | Description                                           | Shape                       |
| -------- | ----------------------------------------------------- | --------------------------- |
| action   | Collected real-world action data                      | [episode_len, 7]            |
| cam_high | Image frames captured by RGB cameras named 'cam_high' | [episode_len,  480, 640, 3] |
| qpos     | Proprioceptive data of robot arm (joint angle)        | [episode_len, 7]            |



#### Training

- ```./vla-scripts/finetune_real_world.py```: Finetune UniVLA on real-world data



```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 finetune_real_world.py  \
         --batch_size 4 \       # Adjust based on your compute setup
         --grad_accumulation_steps 2 \    # A workaround for larger equivalent batch size
         --max_steps 10000 \    # Number of training steps, adjust based on your data volume
         --save_steps 2500 \    # Steps to save intermediate ckpts
         --window_size 10 \     # Frames interval for LAM, also the action chunk size, adjust based on your data frequency
         --run_root_dir "./real-world-log" 
```



### Inference

Once finished fine-tuning of UniVLA and get the action decoder head tailored to your embodiment action space, let's deploy it and see how it works!

Here we provide a deployment example ```./vla-scripts/real_world_employment.py``` which is mainly about `UniVLAInference` class.


> [!NOTE]
> Due to differences in deployment code among various embodiments, we present a general example below for reference. Action chunking is also implemented within the `UniVLAInference` class.

```python
# Register UniVLA 
policy = UniVLAInference(saved_model_path=saved_model_path, pred_action_horizon=12, decoder_path=decoder_path)

# curr_image is read from a camera
resized_curr_image = torchvision.transforms.Resize((224, 224))(torch.flip(curr_image[0],(1,))) # Resize + BGR2RGB (not necessary)

# create fake inputs
task_instruction = 'Store the screwdriver'
proprio = torch.zeros((1,7))

# sample actions
all_actions = policy.step(resized_curr_image, task_instruction, proprio)
```

