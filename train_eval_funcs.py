import numpy as np
import torch
import pandas as pd
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from models import *


def eval_epoch(model, train_for_val, criterion, val_part, seq_len=256, num_clusters=8, device=default_device):
    model.eval()
    y_loss_batches = []
    y_r2_batches = []
    split_idx = int(seq_len * (1 - val_part))

    # маска для правильного вычисления весов (чтобы последние val_part примеров не учитывались при вычислении весов)
    predict_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    for j in range(seq_len):
        predict_mask[j, j] = True
        predict_mask[j, split_idx:] = True

    for i in range(len(train_for_val[0])):  # итерация по всем батчам данных из val_batches
        X, y, t = train_for_val[0][i], train_for_val[1][i], train_for_val[2][i] # уже после FCNet
        X = X.to(device)
        y = y.to(device)
        t = t.to(device)
        with torch.enable_grad():
            y_pred, yc_pred, cate_pred = model.forward(X, y, t, 1 - t, predict_mask.to(device))
            y_loss = torch.mean(criterion(y_pred[:, split_idx:], y[split_idx:].unsqueeze(0)))  # средний лосс по батчу, unsqueeze(0) - чтобы совпадали размерности (1 * n)
            y_r2 = compute_r2_score(y_pred[:, split_idx:].squeeze(0), y[split_idx:])
            y_loss_batches.append(y_loss.item())  # присоединяем среднее по батчу
            y_r2_batches.append(y_r2)

    return np.mean(y_loss_batches), np.mean(y_r2_batches)

# copied from huggingface
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def compute_r2_score(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    return r2_score(y_true_np, y_pred_np)

def fit_epoch(model, train_batches, criterion, optimizer, mask, batch_size, device=default_device):
    model.train()
    y_losses = []
    y_r2_scores = []

    for i in range(train_batches[0].shape[0]):  # итерация по всем батчам данных из train_batches
        X_train, y_train, t_train = train_batches[0][i], train_batches[1][i], train_batches[2][i]
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        t_train = t_train.to(device)

        # Замена значений в treat_train: 0 -> 1, 1 -> 0
        t_train_inverted = 1 - t_train

        optimizer.zero_grad()

        # y_cfactual_pred - предсказание y_cfactual (какое бы было 'y' у конкретного примера, если бы treatment было противоположным)
        # cate - conditional treatment effect
        y_pred, yc_pred, cate_pred = model.forward(X_train, y_train, t_train, t_train_inverted, mask.to(device))
        y_pred = y_pred.squeeze(0)
        nan_mask = ~torch.isnan(y_pred)
        #print(torch.sum(nan_mask).item())
        y_pred = y_pred[nan_mask]
        y_train = y_train[nan_mask]

        if len(y_train) == 0:
            print('All predictions are NaNs, skipping this batch')
            return None, None

        y_loss = criterion(y_pred, y_train)  # unsqueeze(0) - чтобы совпадали размерности (1 * n)
        y_r2 = compute_r2_score(y_pred, y_train)
        y_losses.append(y_loss.item())  # присоединяем среднее по батчу
        y_r2_scores.append(y_r2)

        #nan_mask_loss = torch.isnan(y_loss)
        #print("y_loss nans", torch.sum(nan_mask_loss).item())

        torch.mean(y_loss).backward()  # шаг градиента по y_loss

        optimizer.step()
        print(
            f"batch_number: {i + 1}, y_loss: {y_loss.item()}, y_r2: {y_r2}")

    return np.mean(y_losses), np.mean(y_r2_scores)  # средние значения лосса по Y по эпохе


def collect_forward_outputs(model, all_train_dataset, file_path, device=default_device):
    model.eval()

    x_outputs = []
    y_all = []
    t_all = []
    with torch.enable_grad():
        for batch in all_train_dataset:
            x, y, t = batch
            x = x.reshape(-1, x.shape[-1]).to(device)
            x[torch.isnan(x)] = 0.0
            y = y.reshape(-1).to(device)
            t = t.reshape(-1).to(device)

            x_output = model.forward_FCNet(x)
            x_outputs.append(x_output)
            y_all.append(y)
            t_all.append(t)

    x_outputs = torch.cat(x_outputs, dim=0)
    y_all = torch.cat(y_all)#.view(-1, 1)
    t_all = torch.cat(t_all)#.view(-1, 1)

    x_outputs_np = x_outputs.detach().cpu().numpy()
    y_all_np = y_all.detach().cpu().numpy()
    t_all_np = t_all.detach().cpu().numpy()

    df_x_outputs = pd.DataFrame(x_outputs_np)
    df_y_all = pd.DataFrame(y_all_np, columns=['y'])
    df_t_all = pd.DataFrame(t_all_np, columns=['t'])

    # Объединение всех DataFrame по горизонтали
    df = pd.concat([df_x_outputs, df_y_all, df_t_all], axis=1)

    # Запись DataFrame в CSV файл
    df.to_csv(file_path, index=False)

    return (x_outputs, y_all, t_all)


def train(writer, model, optimizer, scheduler=get_cosine_schedule_with_warmup, num_epochs=101, batch_size=512, valid_step=10,
          max_num_features=10, seq_len=512, lr=0.01, warmup_epochs=10, weight_decay=0.0, mahalanobis=False):
    model.train()
    log_template = "\nEpoch {epoch:03d} train_y_loss: {train_y_loss:0.4f} train_r2: {train_r2:0.4f} val_y_loss: {val_y_loss:0.4f} val_r2: {val_r2:0.4f}"
    history = []
    cur_model = None
    cur_loss = float('inf')
    cur_r2 = float('-inf')
    num_incr_val_loss = 0  # количество эпох, в течение которых повышается val_loss
    with tqdm(desc="epoch", total=num_epochs) as pbar_outer:
        criterion = nn.MSELoss()
        #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = scheduler(optimizer, warmup_epochs,
                              num_epochs if num_epochs is not None else 100)  # when training for fixed time lr schedule takes 100 steps
        for epoch in range(num_epochs):
            config = get_prior_config('causal')
            config_sample = evaluate_hypers(config)
            params = {key: value for key, value in config_sample.items() if key != list(config_sample.keys())[-1]}
            diff_params = config_sample[list(config_sample.keys())[-1]]
            params = get_mlp_prior_hyperparameters(params)
            num_features = random.randint(1, max_num_features)
            x, y, y_, t, packed_hyperparameters = get_batch(batch_size=batch_size, seq_len=seq_len,
                                                            num_features=num_features, get_batch=get_batch_mlp,
                                                            differentiable_hyperparameters=diff_params,
                                                            hyperparameters=params, batch_size_per_gp_sample=1)
            val_part = 0
            train_size = batch_size
            if epoch % valid_step == 0:
                val_part = 0.3
                train_size = int((1 - val_part) * train_size)
            x_stand = normalize_data(x, train_size).permute(1, 0, 2)
            y = y.permute(1, 0)
            t = t.permute(1, 0)
            x_stand[torch.isnan(x_stand)] = 0.0
            y[torch.isnan(y)] = 0.0
            t[torch.isnan(t)] = 0.0
            since = time.time()
            # print(x_stand.shape, y.shape, t.shape)
            train_batches = (x_stand, y, t)

            # mask = torch.eye(train_size, dtype=bool)  # True на диагонали
            mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
            for j in range(seq_len):
                mask[j, j] = True
                mask[j, train_size:] = True
            train_y_loss, train_r2 = fit_epoch(model, train_batches, criterion, optimizer, mask, train_size)
            if train_y_loss is None and train_r2 is None:
                print(f'Skip epoch {epoch} because of NaNs')
                return

            val_y_loss = float('inf')
            val_r2 = float('-inf')
            if epoch % valid_step == 0:
                val_y_loss, val_r2 = eval_epoch(model, train_batches, criterion, val_part, seq_len)
                writer.add_scalar('MSE/train', train_y_loss, epoch)
                writer.add_scalar('MSE/val', val_y_loss, epoch)
                writer.add_scalar('R2/train', train_r2, epoch)
                writer.add_scalar('R2/val', val_r2, epoch)
                if val_y_loss > cur_loss:
                    num_incr_val_loss += 1
                    if num_incr_val_loss >= 10:
                        print(f"Validation loss increased for 10 epochs. Stopping training. Current epoch: {epoch}.")
                        break
                else:
                    num_incr_val_loss = 0
                    cur_model = model  # сохраняем модель только при уменьшающемся лоссе
                cur_loss = val_y_loss
                cur_r2 = val_r2

                '''if val_y_loss < best_loss and val_r2 > best_r2: # количество эпох дб кратным 10, чтобы могла выбираться лучшая модель
                    best_loss = val_y_loss
                    best_r2 = val_r2
                    best_model = model
                '''

            history.append((train_y_loss, train_r2, val_y_loss, val_r2))

            scheduler.step()  # "затухание" lr оптимизатора

            # отображение статуса обучения
            pbar_outer.update(1)

            log_template = "\nEpoch {ep}: Train Y MSE: {train_y_loss}, Train Y R2: {train_r2}, Val Y MSE: {val_y_loss}, Val Y R2: {val_r2}"
            tqdm.write(
                log_template.format(ep=epoch + 1, train_y_loss=train_y_loss, train_r2=train_r2, val_y_loss=val_y_loss,
                                    val_r2=val_r2))

            time_elapsed = time.time() - since
            print('Training for one epoch complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

    # get_embeddings(model, all_train_dataset, file_path)
    return (history, cur_loss, cur_r2, cur_model, optimizer_state, scheduler_state)