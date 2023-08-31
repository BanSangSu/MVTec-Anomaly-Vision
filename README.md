# Anomaly Detection with Computer Vision

### **Computer Vision Anomaly Detection Algorithm Competition** <2022.04.01 ~ 2022.05.13> (Dacon)
https://dacon.io/en/competitions/official/235894/overview/description
<br>
Private score(0.84268) 44th. 😋

## Summary
- Model: EfficientNet B3 and B4
- Dataset: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- Schedular: No schedular
- Cross validation: K(5)-fold
- Optimizer: AdamW
- Criterion: Cross Entropy Loss
- Ensemble: 5fold ensemble

## Next
- I want to use different model(like RegNet, EfficientNetV2)
- change Criterion(Focal loss) and optimizer(Lamb)
- use schedular(e.g. WarmUpLR, OneCycleLR)
- make more preprocessing code
- use TTA(Test Time Augmentation)

## Add
- Data in drive: 3526

