from mmengine.fileio import dump, load
from mmengine.logging import MMLogger
from torch.utils.data import Dataset
from mmcls.registry import DATASETS
from pathlib import Path
from tqdm import tqdm

import numpy as np, pandas as pd
import warnings
import torch
import copy
import pandas

@DATASETS.register_module()
class SubwayDataset(Dataset):
    meta = {'column_names': ['collect_time', 'Emergencyunlockingstatus', 'opendoor', 'openclosed',
                             'dooraction', 'lockswitchstate', 'closewtichstate', 'Voltage', 'current',
                             'Antipinchforce', 'speed', 'corner', 'prepra1', 'doorcount', 'opening',
                             'closing', 'opening_command', 'closing_command', 'label'],
            'label_number': [14, 15, 16, 17, 18, 19, 20, 21],
            'norm_names': ['Voltage', 'current', 'Antipinchforce', 'speed', 'corner']
            }
    name_table = {name: i for i, name in enumerate(meta['column_names'])}
    label_table = {number: i for i, number in enumerate(meta['label_number'])}

    def __init__(self, data_path, sample_points=640, min_valid_points=500):
        self.data_path = data_path
        self.data_list = load(data_path)
        self.logger = MMLogger.get_current_instance()
        self.filter_data(min_valid_points)
        self.sample_points = sample_points

    def __getitem__(self, index):
        """
        Return data:
            1. subway data, (sample_points, num_features)
            2. label, int
        """
        raw_data = self.data_list[index]
        data, label = self.process(raw_data)
        return data, label

    def __len__(self):
        return len(self.data_list)

    def filter_data(self, min_valid_points):
        """ Remove sample points where dooraction is 0, and
        if only few valid sample points left, remove this data.
        """
        new_data_list = []
        n = len(self.data_list)
        action_col = self.name_table['dooraction']

        for i in range(n):
            raw_data = self.data_list[i]
            action_mask = raw_data[:, action_col] == 1
            new_data = raw_data[action_mask]
            if len(new_data) > min_valid_points:
                new_data_list.append(new_data)

        self.logger.info(f'Original data samples {len(self.data_list)} -> '
                         f'Filtered data samples: {len(new_data_list)}')
        self.data_list = new_data_list

    def process(self, raw_data: np.ndarray):
        """ Process data with following steps:
        - check if label consistent, and create an int label
        - remove 'collect_time', 'label' column
        - normalize if feature's distribution is not {0, 1}
        - pad or cut data to fixed shape: (self.sample_points, features)
        """
        # create label
        label_col = self.name_table['label']
        label = raw_data[:, label_col]
        if not self.check_label_concsist(label):
            count_col = self.name_table['doorcount']
            count_id = raw_data[0, count_col]
            warnings.warn(f'labels in one sequence do not consist, check id {count_id}')
        label = self.label_table[label[0]]

        # normalize specified dims
        norm_names = self.meta['norm_names']
        norm_cols = [self.name_table[name] for name in norm_names]
        norm_data = raw_data[:, norm_cols].astype(np.float32)
        mean, std = np.mean(norm_data, axis=0), np.std(norm_data, axis=0)
        norm_data = (norm_data - mean) / std
        raw_data[:, norm_cols] = norm_data

        # delete 'collect_time', 'label', 'doorcount'
        time_col = self.name_table['collect_time']
        door_col = self.name_table['doorcount']
        raw_data = np.delete(raw_data, [time_col, label_col, door_col],
                         axis=1).astype(np.float32)

        # make data fixed shape
        points, features = raw_data.shape
        new_data = np.zeros((self.sample_points, features), dtype=np.float32)
        points = min(points, self.sample_points)
        new_data[:points] = raw_data[:points]
        
        return new_data, label

    def check_label_concsist(self, label):
        value = label[0]
        return np.all(label == value)

    def label_distribution(self):
        label_list = []
        n = len(self.data_list)
        for i in range(n):
            _, label = self[i]
            label_list.append(label)
        from collections import Counter
        return Counter(label_list)

    @staticmethod
    def dooraction_counts(dooraction):
        """ How many dooractions have within a doorcount
        dooraction, (N,) {0, 1}
        """
        da_set, check_set = set(dooraction), set([0, 1])
        assert da_set.issubset(check_set), f'dooraction must be subset of {set([0, 1])}'\
                                           f'but have {da_set}'
        n = len(dooraction)
        counts = 0
        for i in range(n):
            if i == 0:
                last = dooraction[i]
                if last == 1: counts += 1
                continue
            if dooraction[i] == last:
                continue
            else:
                if last == 0: counts += 1
                last = dooraction[i]
        return counts

    @staticmethod
    def reverse_label_table(labels):
        """ Reverset labels to origianl label_number
        labels: ndarray or list
        """
        labels = copy.deepcopy(labels)
        assert isinstance(labels, (np.ndarray, list))
        rev_label_tabel = {i: label for label, i in
                           SubwayDataset.label_table.items()}
        def reverse(labels):
            for i, item in enumerate(labels):
                if isinstance(item, (int, np.int_)):
                    labels[i] = rev_label_tabel[labels[i]]
                else: reverse(labels[i])

        reverse(labels)
        return labels

    @staticmethod
    def result_to_csv(label, pred, out_file):
        assert isinstance(pred, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        data = torch.stack((label, pred)).numpy().T
        data = SubwayDataset.reverse_label_table(data)
        data = pd.DataFrame(data, columns=['label', 'pred'])
        data.to_csv(out_file, index=False)

    @staticmethod
    def split_data(csv_file, out_dir, train_ratio=0.9, shuffle=True):
        print('Loading data...')
        data = pandas.read_csv(csv_file)

        print('Spliting index...')
        np.random.seed(0)
        counts = list(set(data['doorcount'].to_numpy()))  # remove duplicate
        if shuffle:
            perm = np.random.permutation(len(counts))
        else:
            perm = np.arange(len(counts))
        counts = np.array(sorted(counts))[perm]           # random permutation

        # data split accroding to train ratio
        n_train = int(len(counts) * train_ratio)
        train_ids = set(counts[:n_train])
        val_ids = set(counts[n_train:])

        train_data = []
        val_data = []
        np_data = data.to_numpy()
        print('Creating test & val set')
        for id in tqdm(counts):
            id_mask = np_data[:, 13] == id
            id_data = np_data[id_mask]
            if id in train_ids:
                train_data.append(id_data)
            elif id in val_ids:
                val_data.append(id_data)

        out_dir = Path(out_dir)
        train_file = out_dir / 'train_data.pkl'
        val_file = out_dir / 'val_data.pkl'
        dump(train_data, train_file)
        dump(val_data, val_file)

if __name__ == '__main__':
    csv_file = '/github/SubwayHealthCls/data/subway/dms1118c_clean.csv'
    out_dir = '/github/SubwayHealthCls/data/subway'
    SubwayDataset.split_data(csv_file, out_dir, train_ratio=1.0, shuffle=False)