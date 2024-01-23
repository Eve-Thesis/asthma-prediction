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

lr1 = [1e-5, 1e-4]
lr2 = [1e-4, 1e-3]
reg_l2 = [1e-7, 1e-3]
dropout_rate = [0.5, 0.6]
trained_layers = [8, 12]
lr_decay = [0.8]
num_units = [64]

# Kombinacje parametrów
param_combinations = itertools.product(lr1, lr2, lr_decay, reg_l2, dropout_rate, num_units, trained_layers)

for params in param_combinations:
    command = ["python", "model_train_vad.py",
               f"--lr1={params[0]}", f"--lr2={params[1]}",
               f"--lr_decay={params[2]}", f"--reg_l2={params[3]}",
               f"--dropout_rate={params[4]}",
               f"--num_units={params[5]}",
               f"--trained_layers={params[6]}"]
    subprocess.run(command)
