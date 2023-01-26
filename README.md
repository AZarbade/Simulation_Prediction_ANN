# WIP
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

---

## 💪 To - Do

- [x] ~Proper experiment logging~
- [x] Report and findings #1
- [ ] [Testing models](https://wandb.ai/wrongcolor/hvis_rtdl_baseline?workspace=user-wrongcolor)
  - [x] [MLP](https://arxiv.org/pdf/2106.11959.pdf)
  - [x] [ResNet](https://arxiv.org/pdf/2106.11959.pdf)
  - [x] [FT-Transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
  - [ ] [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - [ ] [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html)
  - [ ] [TabNet](https://www.aaai.org/AAAI21Papers/AAAI-1063.ArikS.pdf)
  - [ ] [TabTransformers](https://arxiv.org/abs/2012.06678)
  <!-- - [ ] [GrowNet](https://arxiv.org/abs/2002.07971) -->
  <!-- - [ ] [Tree Ensemble Layers](https://arxiv.org/abs/2002.07772v2) -->
  <!-- - [ ] [Self Normalizing NN](https://arxiv.org/abs/1706.02515v5) -->
  <!-- - [ ] [Neural Oblivious Decision Ensembles](https://arxiv.org/abs/1909.06312) -->
  <!-- - [ ] [AutoInt](https://arxiv.org/abs/1810.11921v2) -->
  <!-- - [ ] [Deep and Cross NN](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754) -->

---
