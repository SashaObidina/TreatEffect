import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler
from train_eval_funcs import *
from generation import *

if __name__ == '__main__':
    from sklearn.model_selection import ParameterSampler
    import numpy as np
    import pickle

    param_grid = {
        'p': uniform(0.1, 0.5),
        'linear1_output': [128, 256, 512, 1024],
        'linear2_output': [128, 256, 512, 1024],
        'linear3_output': [16, 32, 64, 128, 256],
        'num_heads': [4, 8],
        'num_layers': [6, 12]
    }

    best_model_mse = None
    best_loss_mse = float('inf')
    best_r2_mse = float('-inf')
    best_history_mse = []
    best_params_mse = {}

    best_model_r2 = None
    best_loss_r2 = float('inf')
    best_r2_r2 = float('-inf')
    best_history_r2 = []
    best_params_r2 = {}

    n_iter = 2
    param_sampler = ParameterSampler(param_grid, n_iter=n_iter)  # случайно сгенерированные комбинации параметров

    for i, params in enumerate(param_sampler):
        print(f"Iteration {i + 1}/{n_iter}: Hyperparameters: {params}")
        model = ModelExp(input_dim=100, p=params['p'], linear1_output=params['linear1_output'],
                         linear2_output=params['linear2_output'], linear3_output=params['linear3_output'],
                         num_heads=params['num_heads'], num_layers=params['num_layers'])
        history, mse, r2, model, optimizer_state, scheduler_state = train(model=model,
                                                                          scheduler=get_cosine_schedule_with_warmup,
                                                                          num_epochs=3, batch_size=64, valid_step=10,
                                                                          max_num_features=10, seq_len=256, lr=0.01,
                                                                          warmup_epochs=10, weight_decay=0.0,
                                                                          mahalanobis=True)

        filename = 'model_optimizer_scheduler_{}.pkl'.format(i)
        data_to_save = (model, optimizer_state, scheduler_state)
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)

        # построение графика
        epoch_points = list(range(0, len(history) + 1, 10))
        plt.figure(figsize=(12, 6))
        print(len(history))

        history_points = [history[i] for i in epoch_points]
        train_losses = [point[0] for point in history_points]
        val_losses = [point[2] for point in history_points]
        train_r2 = [point[1] for point in history_points]
        val_r2 = [point[3] for point in history_points]

        # MSE plot
        plt.subplot(1, 2, 1)
        plt.plot(epoch_points, train_losses, 'b-', label='Train MSE')
        plt.plot(epoch_points, val_losses, 'r-', label='Validation MSE')
        plt.title('MSE vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.xticks(epoch_points)
        plt.legend()

        # R2 plot
        plt.subplot(1, 2, 2)
        plt.plot(epoch_points, train_r2, 'b-', label='Train R2')
        plt.plot(epoch_points, val_r2, 'r-', label='Validation R2')
        plt.title('R2 vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('R2')
        plt.xticks(epoch_points)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'iteration_{i}_plot.png')
        plt.show()

        if mse < best_loss_mse:  # and r2 > best_r2:
            best_loss_mse = mse
            best_r2_mse = r2
            best_model_mse = model
            best_history_mse = history
            best_params_mse = params

        '''if r2 > best_r2_r2:
            best_loss_r2 = mse
            best_r2_r2 = r2
            best_model_r2 = model
            best_history_r2 = history
            best_params_r2 = params
        '''

    print("best_loss (mse comparing):", best_loss_mse)
    print("best_r2 (mse comparing):", best_r2_mse)
    print("best_params (mse comparing):", best_params_mse)
    print("best_history (mse comparing)", best_history_mse)
