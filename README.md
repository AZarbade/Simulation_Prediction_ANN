<h1 align="center">Simulation Results Prediction using Neural Networks</h1>
<h3 align="center">Use of Neural Networks in Analysis and Simulation field</h3>

---
## About This Project
- 👓 Supervised and developed under **Armament Research & Development Establishment, Pashan, Pune 411021, India**
- 🔭 Checking to see if using Neural Networks are capable of aiding in Analysis and Simulation fields
- 🔮 Predicting penetration index of ballistics on different materials, but using only limited simulation data
- 🌱 If sucessful this project can greatly reduce the time taken in any major development process
- 🏗️ Made with 💖 using <img height="16" width="16" src="https://cdn.simpleicons.org/pytorch" style="vertical-align: bottom;"/>

---

## 🛠 Current Implementation

- **[Multi Layer Perceptron - MLP](LINK)** - A simple MLP neural network is being used to predict the results. MLP [baseline](https://wandb.ai/wrongcolor/HVIS_Baseline?workspace=user-wrongcolor), here Loss: mean_squared_error and root_mean_squared_error.. Hyper-parmeters were tunned to minimize the validation loss. K-Fold method is used to circulate data points and average out scores over 15 runs.
- **[Impact of Pre-Processing](https://wandb.ai/wrongcolor/HVIS_PreProcessingCheck?workspace=user-wrongcolor)** - Detailed view and logs for checking **impact of pre-processing on loss**. Here, Loss: mean_squared_error and root_mean_squared_error.
- **[Using RTDL library](https://github.com/Yura52/rtdl)** - Propsed in [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959), this library contains 3 basic neural network implementations to start working on top of.

## 📊 Stats

| Model           | RMSE   |
| --------------- | ------ |
| **MLP**         | 0.1045 |
| **ResNet**      | 0.0972 |
| **FT-T**        | 0.0473 |
| **SNN**         | 0.9773 |
| **NODE**        | 0.0424 |
| **XGBoost**     | 0.0462 |
<!-- | **GrowNet**     | 0.487 | -->
<!-- | **DCN2**        | 0.484 | -->
<!-- | **TabNet**      | 0.510 | -->

---

## 💪 To - Do

- [x] ~Proper experiment logging~
- [x] Report and findings #1
- [ ] [Testing models](https://wandb.ai/wrongcolor/hvis_rtdl_baseline?workspace=user-wrongcolor)
  - [x] [MLP](https://arxiv.org/pdf/2106.11959.pdf)
  - [x] [ResNet](https://arxiv.org/pdf/2106.11959.pdf)
  - [x] [FT-Transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
  - [x] [SNN](https://arxiv.org/pdf/1706.02515.pdf)
  - [x] [NODE](https://arxiv.org/pdf/1909.06312.pdf)
  - [ ] [TabNet]()
  - [ ] [GrowNet]()
  - [ ] [DCN V2]()
  - [ ] [AutoInt]()
  - [ ] [XGBoost]()
  - [ ] [CatBoost]()

---
