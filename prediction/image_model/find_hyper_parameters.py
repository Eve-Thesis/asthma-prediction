import itertools
import subprocess

# lr1 = [1e-5, 1e-4, 1e-3]
# lr2 = [1e-4, 1e-3, 1e-2]
# lr_decay = [0.8, 0.9, 0.99]
# reg_l2 = [1e-7, 1e-5, 1e-3]
# dropout_rate = [0.3, 0.5, 0.7]
# num_units = [32, 64, 128, 256]
# trained_layers = [2, 6, 10]

# lr1 = [1e-5, 1e-3]
# lr2 = [1e-4, 1e-2]
# lr_decay = [0.8, 0.99]
# reg_l2 = [1e-7, 1e-3]
# dropout_rate = [0.3, 0.7]
# num_units = [32, 64, 128]
# trained_layers = [2, 10]

# lr1 = [1e-5, 1e-3]
# lr2 = [1e-4, 1e-2]
# reg_l2 = [1e-7, 1e-3]
# dropout_rate = [0.3, 0.7]
# trained_layers = [2, 10]
# lr_decay = [0.8]
# num_units = [64]

lr_base = [1e-6, 1e-4]
# lr1 = [1e-5, 1e-4]
# lr2 = [1e-4, 1e-3]
reg_l2 = [1e-5, 1e-3]
dropout_rate = [0.3, 0.5]
trained_layers = [16, 32]
lr_decay = [0.8]
num_units = [64, 128]
model_name = ['resnet', 'densenet']

param_combinations = itertools.product(lr_base, lr_decay, reg_l2, dropout_rate, num_units, trained_layers, model_name)

for params in param_combinations:
    command = ["python", "train_pipe.py",
               f"--lr1={params[0]}", f"--lr2={params[0]*10}",
               f"--lr_decay={params[1]}", f"--reg_l2={params[2]}",
               f"--dropout_rate={params[3]}",
               f"--num_units={params[4]}",
               f"--trained_layers={params[5]}",
               f"--model_type={params[6]}"]
    subprocess.run(command)
