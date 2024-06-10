import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def estimate_feature_importance_with_t(X, y, t, n_top_features=10):
    #X_with_t = np.hstack((X, t.reshape(-1, 1)))

    model = RandomForestRegressor()
    model.fit(X, y)
    feature_importances = model.feature_importances_

    sorted_indices = np.argsort(feature_importances)[::-1]
    top_feature_indices = sorted_indices[:n_top_features]
    top_features = [f"Feature {i}" for i in top_feature_indices]

    return top_feature_indices


class IHDP(object):
    def __init__(self, path_data="IHDP/csv/", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]
        self.num_features = 25

    def load_data(self):
        all_data = []
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            all_data.append(data)
        all_data = np.vstack(all_data)
        return all_data

    def get_train_test(self):
        data = self.load_data()
        data[:, 0] = data[:, 0].astype(int)  # Convert boolean to int
        ihdp_df_new = data.copy()

        # Разделение на признаки и целевые переменные
        t = ihdp_df_new[:, 0].astype(np.int32)
        x = ihdp_df_new[:, 5:].astype(np.float32)
        mu_0 = ihdp_df_new[:, 3].astype(np.float32)#[:, np.newaxis]
        mu_1 = ihdp_df_new[:, 4].astype(np.float32)#[:, np.newaxis]
        Y0 = (mu_0 - np.mean(mu_0)) / np.std(mu_0)
        Y1 = (mu_1 - np.mean(mu_1)) / np.std(mu_1)
        cate = Y1 - Y0

        # Преобразуем бинарный признак, который в {1, 2}, в {0, 1}
        x[:, 13] -= 1

        mask_cnt, mask_trt = t == 0, t == 1
        Y = np.empty_like(Y0)
        Y[mask_cnt] = Y0[mask_cnt]
        Y[mask_trt] = Y1[mask_trt]

        top_features_indices = estimate_feature_importance_with_t(x, Y, t, n_top_features=10)
        x = x[:, top_features_indices]

        # Разделение на train, valid и test
        itr, ite = train_test_split(np.arange(x.shape[0]), test_size=0.3, random_state=1, stratify=t)  # train, test
        scaler = StandardScaler()
        # Вычислить средние и стандартные отклонения обучающего набора данных и применить их к данным
        X_train_scaled = scaler.fit_transform(x[itr])
        X_test_scaled = scaler.transform(x[ite])



        # Преобразование в тензоры PyTorch
        train = (torch.tensor(x[itr], dtype=torch.float32),
                      torch.tensor(t[itr], dtype=torch.float32)), \
                     (torch.tensor(Y[itr], dtype=torch.float32), )

        test = (torch.tensor(X_test_scaled, dtype=torch.float32),
                      torch.tensor(t[ite], dtype=torch.float32)), \
                     (torch.tensor(Y0[ite], dtype=torch.float32),
                      torch.tensor(Y1[ite], dtype=torch.float32),
                      #torch.tensor(Y[ite], dtype=torch.float32),
                      torch.tensor(cate[ite], dtype=torch.float32), )

        # Количество элементов в группах контроля и лечения
        n_control = np.sum(t[itr] == 0)
        print('control in train:', n_control)
        n_treatment = np.sum(t[itr] == 1)
        print('treat in train:', n_treatment)

        n_control = np.sum(t[ite] == 0)
        print('control in control:', n_control)
        n_treatment = np.sum(t[ite] == 1)
        print('treat in control:', n_treatment)

        return train, test
