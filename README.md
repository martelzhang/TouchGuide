# TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance

![insight](images/insight.png)

Project Page: [martelzhang.github.io/touchguide](https://martelzhang.github.io/touchguide)

Zhemeng Zhang, Jiahua Ma, Xincheng Yang, Xin Wen, Yuzhi Zhang, Boyan Li, Yiran Qin, Jin Liu, Can Zhao, Li Kang, Haoqin Hong, Zhenfei Yin, Philip Torr, Hao Su, Ruimao Zhang, Daolin Ma.

Shanghai Jiao Tong University, Xense Robotics, Sun Yat-sen University, University of Oxford, Shanghai AI Laboratory, Shanghai AI Laboratory, University of California San Diego.

# TODO List
- [ ] **TacUMI** harware.
- [x] Data collection code.
- [ ] **TouchGuide** code.

# Environment Setup
We provide convenient scripts to help you quickly install all the required environments for **TouchGuide** and **TacUMI** (data collection).

```bash
cd data_collection
```

You can install using either **conda** or **mamba** (whichever is more convenient). You only need to choose one of the two environment managers.


```bash
./setup_env.sh --conda
```
or
```bash
./setup_env.sh --mamba
```

Then install **XenseSDK**, **ARX5 SDK**, and all other required packages.

```bash
./setup_env.sh --install
```

# Data Collection

We use the **LeRobot** format for data collection. Likewise, we provide a unified and efficient wrapper. For **ARX5**, use the following command for data collection:

```bash
lerobot-teleoperate \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --fps=30 \
    --debug_timing=false \
    --display_data=true
```

For **TacUMI**, you can use the following command for data collection:

```bash
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=<your_mac_address> \
    --dataset.repo_id=<your_repo_id> \
    --dataset.num_episodes=20 \
    --dataset.single_task="your prompt" \
    --dataset.fps=30 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

# Citation
If you find this work helpful, we would greatly appreciate it if you cite our paper.
```
@misc{zhang2026touchguide,
title={TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance},
author={Zhemeng Zhang and Jiahua Ma and Xincheng Yang and Xin Wen and Yuzhi Zhang and Boyan Li and Yiran Qin and Jin Liu and Can Zhao and Li Kang and Haoqin Hong and Zhenfei Yin and Philip Torr and Hao Su and Ruimao Zhang and Daolin Ma},
year={2026},
eprint={2601.20239},
archivePrefix={arXiv},
primaryClass={cs.RO},
url={https://arxiv.org/abs/2601.20239},
}
```