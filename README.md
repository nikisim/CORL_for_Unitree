# CORL for Unitree and Fetch tasks (Clean Offline Reinforcement Learning)

🧵 CORL is an Offline Reinforcement Learning library that provides high-quality and easy-to-follow single-file implementations of SOTA ORL algorithms. Each implementation is backed by a research-friendly codebase, allowing you to run or tune thousands of experiments. Heavily inspired by [cleanrl](https://github.com/vwxyzjn/cleanrl) for online RL, check them out too!<br/>

* 📜 Single-file implementation
* 📈 Benchmarked Implementation for N algorithms
* 🖼 [Weights and Biases](https://wandb.ai/site) integration
------
* ⭐**new**⭐ Preloaded Datasets with > 700.000 iterations in D4RL format
* ⭐**new**⭐ Model Saving
* ⭐**new**⭐ Video Saving for Fetch Tasks


## Datasets
Datasets are available from [Google Disk](https://drive.google.com/drive/folders/1f4EEHk_lzOkLD_f_XL6ypsmIumbzRlq1?usp=sharing)

Current datasets:

| Environment | Dataset | Pretrained model| GIF
| -------------| ------ |------ | ---- |
| FetchReach   |✅      |ReBRAC ✅, IQL ❌| <img src="Images/Fetch_REACH.gif" width="160" height="106" />|
| FetchPush   |✅      |ReBRAC ✅, IQL ❌|<img src="Images/Fetch_PickAndPlace.gif" width="160" height="106" />|
| FetchPickAndPlace   |✅      |ReBRAC ✅, IQL ❌| <img src="Images/Fetch_PickAndPlace.gif" width="160" height="106" />|
| FetchSlide   |✅      |ReBRAC ✅, IQL ❌| <img src="Images/Fetch_SLIDE.gif" width="160" height="106" />|
| Unitree A1 ground task ([MetaGym](https://github.com/PaddlePaddle/MetaGym/tree/master/metagym/quadrupedal))  |✅      |ReBRAC ✅, IQL ❌| <img src="Images/UnitreeA1_Ground.gif" width="160" height="106" />|
<!-- | Unitree A1 StairStair task |🔜      |ReBRAC 🔜| | -->


All Fetch Tasks datasets collected by DDPG+HER from [original repo](https://github.com/TianhongDai/hindsight-experience-replay/tree/master).

Unitree A1 dataset collected by [ETG RL](https://github.com/PaddlePaddle/PaddleRobotics/tree/main/QuadrupedalRobots/ETGRL).

**Be aware that Fetch envs works with Gymnasium now, not Gym!
But Unitree A1 task still use Gym.**

### Fancy Wandb [report](https://api.wandb.ai/links/itmo449/uclfygvm) about ReBRAC usage with datasets above.
(maybe not so fancy, but it will be soon)


## Model evaluation
To evaluate saved model:
```
python3 test_eval.py --env_name FetchPush --config_path data/saved_models/FetchPush/config.json --model_path data/saved_models/FetchPush/actor_state999.pkl --num_episodes 20
```

## Getting started

Docker (Doesn't work yet, use Anaconda)

```bash
git clone https://github.com/tinkoff-ai/CORL.git && cd CORL
pip install -r requirements/requirements_dev.txt

# alternatively, you could use docker
docker build -t <image_name> .
docker run --gpus=all -it --rm --name <container_name> <image_name>
```



