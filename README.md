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
- **[Parameter Domain Reduction and its Imapct](https://wandb.ai/wrongcolor/param_domain?workspace=user-wrongcolor)** - Main purpose of this study is to reduce simulation data overhead. Initially, a reduced parameter domain is used to train and study the models.

## 📊 Stats
<!-- 
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


<!-- Comparisions <img src="./reports/helpers/W&B%20Chart%203_3_2023,%2011_38_42%20am.svg"> -->

<div style="display: flex; align-items: center;">
  <div style="flex: 1;">
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>RMSE</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>MLP</strong></td>
          <td>0.1045</td>
        </tr>
        <tr>
          <td><strong>ResNet</strong></td>
          <td>0.0972</td>
        </tr>
        <tr>
          <td><strong>FT-T</strong></td>
          <td>0.0473</td>
        </tr>
        <tr>
          <td><strong>SNN</strong></td>
          <td>0.9773</td>
        </tr>
        <tr>
          <td><strong>NODE</strong></td>
          <td>0.0424</td>
        </tr>
        <tr>
          <td><strong>XGBoost</strong></td>
          <td>0.0462</td>
        </tr>
      </tbody>
    </table>
  </div>
  <div style="flex: 1; margin-left: -300px;">
    <img src="./reports/helpers/W&B%20Chart%203_3_2023,%2011_38_42%20am.svg" style="width: 100%;">
  </div>
</div>


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
  - [x] [XGBoost]()
  - [ ] [GrowNet]()
  - [ ] [DCN V2]()
  - [ ] [AutoInt]()
  - [ ] [TabNet]()
  - [ ] [CatBoost]()
- [ ] Hyper Parameter Tunning for each model
---
