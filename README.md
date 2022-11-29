# 图神经网络

## 环境

```
dgl==0.9.1
dgl_cu113==0.9.1.post1
numpy==1.21.2
torch==1.10.0+cu113
```
dgl安装：https://www.dgl.ai/pages/start.html

## ChebNet

```bash
python main.py --model ChebNet --cheb-order 4
```

## GCN

```bash
python main.py --model GCN
```

## GraphSAGE

```bash
python main.py --model GraphSAGE --epochs 100 --lr 0.01 --hidden-size 32 --weight-decay 0.0005
```

## GAT

```bash
python main.py --model GAT --epochs 100 --hidden-size 8 --weight-decay 0.0005
```

## JKNet

```bash
python main.py --model JKNet --epochs 200 --weight-decay 0.0005 --lr 0.01
```