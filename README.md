# Image Classification on CIFAR-10 using Deep Learning Neural Networks

Drexel University, Philadelphia, PA

Course: CS615 - Deep Learning

Date: September 2, 2022

Team member:

| Name | Maissoun Ksara | Khiem Nguyen | Tien Nguyen | Chris Oriente |
| --- | --- | --- | --- | --- |
| Email | mk3272@drexel.edu | ktn44@drexel.edu | thn44@drexel.edu | co449@drexel.edu |

## Install dependencies

Python3 and Pip is required to install the dependencies.

```bash
pip install -r requirements.txt
```

## Train and evaluate the Multi-layer Perceptron Model

```bash
python3 mlp_allbatches
```

## Train and evaluate the Convolutional Neural Network (LeNet) Model 1

```bash
python3 lenet.py
```

The result cross correlation vs epochs file (lenet_10000_defaultmodel.png) is placed in the folder lenet_figures

## Train and evaluate the Convolutional Neural Network (LeNet) Model 2

```bash
python3 lenet_dropout.py
```

The result cross correlation vs epochs file (lenet_50000_defaultmodel.png) is placed in the folder lenet_figures