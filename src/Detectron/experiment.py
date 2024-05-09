from __future__ import annotations

from typing import Optional
from warnings import warn

from .record import DetectronRecordsManager
from .ensemble import DetectronEnsemble, BaseModelManager, DatasetsManager
from .strategies import DisagreementStrategy, DetectronStrategy

class DetectronResult:
    """
    A class to store the results of a Detectron test
    """

    def __init__(self, cal_record: DetectronRecordsManager, test_record: DetectronRecordsManager):
        """
        :param cal_record: Result of running benchmarking.detectron_test_statistics using IID test data
        :param test_record: Result of running benchmarking.detectron_test_statistics using the unknown test data
        """
        self.cal_record = cal_record
        self.test_record = test_record

    def calibration_trajectories(self):
        rec = self.cal_record.get_record()
        return rec[['seed', 'model_id', 'rejection_rate']]

    def test_trajectory(self):
        rec = self.test_record.get_record()
        return rec[['model_id', 'rejection_rate']]

    def get_experiments_results(self, strategy : DetectronStrategy, significance_level):
        return(strategy.execute(self.cal_record, self.test_record, significance_level))

    def evaluate_detectron(self, strategy:DetectronStrategy, significance_level):
        return(strategy.evaluate(self.cal_record, self.test_record, significance_level))
    
class DetectronExperiment:
    def __init__(self) -> None:
        pass
    
    
    def run(    self,
                datasets: DatasetsManager,
                training_params: dict,
                base_model_manager: BaseModelManager,
                samples_size : int = 20,
                detectron_result: DetectronResult = None,
                ensemble_size=10,
                num_calibration_runs=100,
                patience=3,
                significance_level=0.1, 
                test_strategy=DisagreementStrategy,
                evaluate_detectron=False, 
                allow_margin : bool = False, 
                margin = 0.05):
        
        print(detectron_result)
        # create a calibration ensemble
        calibration_ensemble = DetectronEnsemble(base_model_manager, ensemble_size)
        # create a testing ensemble
        testing_ensemble = DetectronEnsemble(base_model_manager, ensemble_size)
        # ensure the reference set is larger compared to testing set
        reference_set = datasets.get_reference_data(return_instance=True)
        if detectron_result is None:
            if reference_set is not None:
                test_size = len(reference_set)
                assert test_size > samples_size, \
                    "The reference set must be larger than the testing set to perform statistical bootstrapping"
                if test_size < 2 * samples_size:
                    warn("The reference set is smaller than twice the testing set, this may lead to poor calibration")

                # evaluate the calibration ensemble
                cal_record = calibration_ensemble.evaluate_ensemble(datasets=datasets, 
                                                                    n_runs=num_calibration_runs,
                                                                    samples_size=samples_size, 
                                                                    training_params=training_params, 
                                                                    set='reference', 
                                                                    patience=patience, 
                                                                    allow_margin=allow_margin,
                                                                    margin=margin)
                
                test_record = testing_ensemble.evaluate_ensemble(datasets=datasets, 
                                                                n_runs=100, 
                                                                samples_size=samples_size, 
                                                                training_params=training_params,
                                                                set='testing', 
                                                                patience=patience,
                                                                allow_margin=allow_margin,
                                                                margin=margin)

        else:
            print('provided tesults')
            cal_record = detectron_result.cal_record
            test_record = detectron_result.test_record
            assert cal_record.sample_size == test_record.sample_size, \
                "The calibration record must have been generated with the same sample size as the observation set"
            
    
        # save the detectron runs results
        detectron_results = DetectronResult(cal_record, test_record)
        print(detectron_results.cal_record.counts())
        print(detectron_results.test_record.counts())
        # calculate the detectron test
        experiment_results = detectron_results.get_experiments_results(test_strategy, significance_level)
        # evaluate the detectron if needed
        if evaluate_detectron:
            evaluation_results = detectron_results.evaluate_detectron(test_strategy, significance_level)
        else:
            evaluation_results = None
        return detectron_results, experiment_results, evaluation_results
    

