import os

model_name = 'convnet'
num_iters = 1_600_000
dataset = "cifar10"
samples_per_label = 5000
channels = 8
lr = 0.01

python_cmd = f"python run.py --seed 1 --device cuda:0 --lr {lr} --channels {channels} --samples_per_label {samples_per_label} --loss ce --model {model_name} --num_iters {num_iters} --dataset {dataset}"
os.system(python_cmd)
