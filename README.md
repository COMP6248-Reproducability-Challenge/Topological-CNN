# Topological-CNN

# About

 This report attempted to reproduce the findings in the paper ["Topological Convolutional Neural Networks"](https://openreview.net/forum?id=hntbh8Zo1V) submitted in the  NeurIPS 2020 Workshop on Topological Data Analysis and Beyond. The TCNNs and experiments described in the paper were implemented based on the information provided in the orginal report. The ease of implementation as well as reproduced experiment results were used to comment on the reproducibility of the original paper. The filters re-implemented were the KF and CF filters. The Generalisability experiments, Synthetic experiments and Interpretability experiments were also implemented based on the information provided.

# Environment
- PyTorch
- Torchbearer
```
pip install -r requirements.txt
```

# Training
To train models for interpretability, synthetic, and generalizability experiments, go to corresponding experiment directories in Experiment folder and run:
```
python train.py
```
