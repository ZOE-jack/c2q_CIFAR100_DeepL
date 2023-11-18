## Operating system and equipment
    window wsl2-ubuntu-2.0.9.0
    conda environment
    NVIDIA GeForce RTX 2070 - driver 546.01

## Cuda and cudnn
    pytorch-cuda 12.1
    
## Used Packages
    Python 3.9.18
    Pennylane 0.7.0 
    Torch 2.1.0
    torchvision 0.16.0 (for the pretrained resources)
    matplotlib
    numpy

## Check before running the code in transfer.py
    required path changed: line 147, 156, 161(unless you are in the same current directory with transfer.py)
    optioned path changed: line 44, 47

## How to run the code
    Runing the transfer.py in command line, and type your preference for quantum layers(0 or 1) and batch size with space between them.
    
    EX:16 batch size
        For quantum: python transfer.py 1 16
        For classic: python transfer.py 0 16


## The code is based on the reference:
    [1] Andrea Mari, Thomas R. Bromley, Josh Izaac, Maria Schuld, and Nathan Killoran. 
        Transfer learning in hybrid classical-quantum neural networks. Quantum 4, 340 (2020).
    [2] Andrea Mari, Thomas R. Bromley, Josh Izaac, Maria Schuld, and Nathan Killoran. 
        Transfer learning in hybrid classical-quantum neural networks. arXiv:1912.08278 (2019).
    [3] Reference code1: https://github.com/hchsu/QML_CIFAR10
    [4] Reference code2: https://github.com/XanaduAI/quantum-transfer-learning
    [5] Reference code3: https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning/
