from sklearn.model_selection import train_test_split
from datasets.dataset import dataset_loader, list_files
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np


PATH = '../datasets'


class NearestNeighbor:

    def __init__(self, k=1):
        self._Xtr = None
        self._ytr = None
        self._k = k

    def train(self, x_train, y_train):
        print('[INFO] Training...')
        self._Xtr = x_train
        self._ytr = y_train

    def predict(self, x_test):
        num_test = x_test.shape[0]
        y_pred = np.zeros(num_test, dtype=self._ytr.dtype)

        print("[INFO] Prediction...")

        for i in tqdm(range(num_test)):
            distances = np.sum(np.abs(self._Xtr - x_test[i, :]), axis=1)

            if self._k == 1:
                min_idx = np.argmin(distances)
                y_pred[i] = self._ytr[min_idx]

            else:
                min_idxs = np.argsort(distances)[:self._k]

                votes = {}
                for label in self._ytr[min_idxs]:
                    votes[label] = votes.get(label, 0) + 1

                max_votes = 0
                # max_votes_class = None
                for _class, _votes in votes.items():
                    if _votes > max_votes:
                        max_votes = _votes
                        y_pred[i] = _class
                # y_pred[i] = max_votes_class

        return y_pred


if __name__ == '__main__':

    print('[INFO] Load Dataset')
    data_files = list(list_files(PATH))
    data, labels = dataset_loader(data_files, preprocessing=(32, 32), verbose=500)

    data = data.reshape(data.shape[0], 3072)

    train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=0.33, random_state=42)

    print('[INFO]: Features Matrix: {:.1f}MB'.format(float(data.nbytes / 1024 * 1000.0)))

    model = NearestNeighbor(5)
    model.train(train_X, train_y)

    print(classification_report(test_y, model.predict(test_X)))
