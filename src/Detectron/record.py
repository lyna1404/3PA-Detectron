from ..ModelManager.eval_metrics import RocAuc
import pandas as pd
import numpy as np

class DetectronRecord:
    def __init__(self, seed, model_id, original_count) -> None:
        self.seed = seed
        self.model_id = model_id
        self.original_count = original_count

    def update(self, validation_auc, test_auc, predicted_probabilities, count):
        self.validation_auc = validation_auc
        self.test_auc = test_auc
        self.predicted_probabilities = predicted_probabilities
        self.updated_count = count
        self.rejected_samples = self.original_count - self.updated_count
    
    def to_dict(self):
        return {'seed':self.seed, 
                'model_id': self.model_id,
                'validation_auc': self.validation_auc,
                'test_auc': self.test_auc, 
                'rejection_rate': 1 - self.updated_count/ self.original_count,
                'predicted_probabilities': self.predicted_probabilities, 
                'count': self.updated_count, 
                'rejected_count' : self.rejected_samples} 


class DetectronRecordsManager:
    def __init__(self, sample_size):
        self.records = []
        self.sample_size = sample_size
        self.idx = 0
        self.__seed = None

    def seed(self, seed):
        self.__seed = seed


    def update(self, val_data_x, val_data_y, sample_size, model, model_id,
                predicted_probabilities=None, test_data_x=None, test_data_y=None, eval_metric = RocAuc()):
        
        assert self.__seed is not None, 'Seed must be set before updating the record'
        
        record = DetectronRecord(self.__seed, model_id, self.sample_size)
        validation_auc = model.evaluate(val_data_x, val_data_y, ['Auc']).get('Auc')
        testing_auc = model.evaluate(test_data_x, test_data_y, ['Auc']).get('Auc') if test_data_x is not None else float('nan')
        record.update(validation_auc, testing_auc, predicted_probabilities, sample_size)
        self.records.append(record.to_dict())
        self.idx += 1

    def freeze(self):
        self.records = self.get_record()

    def get_record(self):
        if isinstance(self.records, pd.DataFrame):
            return self.records
        else:
            return pd.DataFrame(self.records)

    def save(self, path):
        self.get_record().to_csv(path, index=False)

    @staticmethod
    def load(path):
        x = DetectronRecordsManager(sample_size=None)
        x.records = pd.read_csv(path)
        x.sample_size = x.records.query('model_id==0').iloc[0]['count']
        return x

    def counts(self, max_ensemble_size=None) -> np.ndarray:
        assert max_ensemble_size is None or max_ensemble_size > 0, 'max_ensemble_size must be positive or None'
        rec = self.get_record()
        counts = []
        for i in rec.seed.unique():
            run = rec.query(f'seed=={i}')
            if max_ensemble_size is not None:
                run = run.iloc[:max_ensemble_size + 1]
            counts.append(run.iloc[-1]['count'])
        return np.array(counts)
    
    def rejected_counts(self, max_ensemble_size=None) -> np.ndarray:
        assert max_ensemble_size is None or max_ensemble_size > 0, 'max_ensemble_size must be positive or None'
        rec = self.get_record()
        counts = []
        for i in rec.seed.unique():
            run = rec.query(f'seed=={i}')
            if max_ensemble_size is not None:
                run = run.iloc[:max_ensemble_size + 1]
            counts.append(run.iloc[-1]['rejected_count'])
        return np.array(counts)

    def count_quantile(self, quantile, max_ensemble_size=None):
        counts = self.counts(max_ensemble_size)
        return np.quantile(counts, quantile, method='inverted_cdf')
    
    def rejected_count_quantile(self, quantile, max_ensemble_size=None):
        rejected_counts = self.rejected_counts(max_ensemble_size=max_ensemble_size)
        return np.quantile(rejected_counts, quantile)

