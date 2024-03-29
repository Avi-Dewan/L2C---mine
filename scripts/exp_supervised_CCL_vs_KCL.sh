# CIFAR100
python demo.py --dataset CIFAR100 --model_type vgg --model_name VGG8 --batch_size 1000 --schedule 70 140 --epochs 160 --loss CCL
python demo.py --dataset CIFAR100 --model_type vgg --model_name VGG8 --batch_size 1000 --schedule 70 140 --epochs 160 --loss KCL
python demo.py --dataset CIFAR100 --model_type vgg --model_name VGG8 --batch_size 1000 --schedule 70 140 --epochs 160 --loss CE

# CIFAR10 @ VGG8
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG8 --schedule 30 60 --epochs 70 --loss CCL
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG8 --schedule 30 60 --epochs 70 --loss KCL
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG8 --schedule  30 60 --epochs 70 --loss CE

# CIFAR10 @ VGG16
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG16 --schedule 30 60 --epochs 70 --loss CCL --optimizer SGD --lr 0.1
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG16 --schedule 30 60 --epochs 70 --loss KCL --optimizer SGD --lr 0.1
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG16 --schedule 30 60 --epochs 70 --loss CE  --optimizer SGD --lr 0.1

# CIFAR10 @ ResNet101
python demo.py --dataset CIFAR10 --model_type resnet --model_name ResNet101 --schedule 30 60 --epochs 70 --loss CCL --optimizer SGD --lr 0.1
python demo.py --dataset CIFAR10 --model_type resnet --model_name ResNet101 --schedule 30 60 --epochs 70 --loss KCL --optimizer SGD --lr 0.1
python demo.py --dataset CIFAR10 --model_type resnet --model_name ResNet101 --schedule 30 60 --epochs 70 --loss CE  --optimizer SGD --lr 0.1

# MNIST @ LeNet
python demo.py --loss CCL
python demo.py --loss KCL
python demo.py --loss CE