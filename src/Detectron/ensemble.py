from tqdm import tqdm
from ..ModelManager.BaseModel import BaseModelManager
from ..DatasetManager.Datasets import DatasetsManager
from .record import DetectronRecordsManager
from .stopper import EarlyStopper

import numpy as np
class DetectronEnsemble:

    def __init__(self, base_model_manager: BaseModelManager, ens_size):
            self.base_model_manager = base_model_manager
            self.base_model = base_model_manager.get_instance()
            self.ens_size = ens_size
            self.cdcs = [base_model_manager.clone_base_model() for _ in range(ens_size)]
    
    def evaluate_ensemble(self, 
                          datasets : DatasetsManager, 
                          n_runs : int, 
                          samples_size : int , 
                          training_params : dict, 
                          set : str = 'reference', 
                          patience : int = 3, 
                          allow_margin : bool = False,
                          margin : int = None):
        
        # set up the training, validation and testing sets
        training_data = datasets.get_base_model_training_data(return_instance=True)
        validation_data = datasets.get_base_model_validation_data(return_instance=True)
        if set=='reference':
            testing_data = datasets.get_reference_data(return_instance=True)
        elif set == 'testing':
            testing_data = datasets.get_testing_data(return_instance=True)
        else:
            raise ValueError("The set used to evaluate the ensemble must be either the reference set or the testing set")

        # set up the records manager
        record = DetectronRecordsManager(sample_size=samples_size)
        
        # evaluate the ensemble for n_runs of runs
        for seed in tqdm(range(n_runs), desc='running seeds'):
            # sample the testing set according to the provided sample_size and current seed
            testing_set = testing_data.sample(samples_size, seed)
            # predict probabilities using the base model on the testing set
            base_model_pred_probs = self.base_model.predict(testing_set.get_features(), True)
            # set pseudo probabilities and pseudo labels predicted by the base model
            testing_set.set_pseudo_probs_labels(base_model_pred_probs, 0.5)
            # the base model is always the model with id = 0
            model_id = 0
            # seed the record
            record.seed(seed)
            # update the record with the results of the base model
            record.update(val_data_x=validation_data.get_features(), val_data_y=validation_data.get_true_labels(), 
                          sample_size=samples_size, model=self.base_model, model_id=model_id, 
                          predicted_probabilities=testing_set.get_pseudo_probabilities(), 
                          test_data_x=testing_set.get_features(), test_data_y=testing_set.get_true_labels())
            # set up the Early stopper
            stopper = EarlyStopper(patience=patience, mode='min')
            stopper.update(samples_size)
            # Initialize the updated count
            updated_count = samples_size
            # Train the cdcs
            for i in range(1, self.ens_size + 1):
                # get the current cdc
                cdc = self.cdcs[i-1]
                # save the model id
                model_id = i
                # update the training params with the current seed which is the model id
                training_params.update({'seed':i})
                # train this cdc to disagree
                cdc.train_to_disagree(x_train=training_data.get_features(), y_train=training_data.get_true_labels(), 
                                      x_validation=validation_data.get_features(), y_validation=validation_data.get_true_labels(), 
                                      x_test=testing_set.get_features(), y_test=testing_set.get_pseudo_labels(),
                                      training_parameters=training_params,
                                      balance_train_classes=True, 
                                      N=updated_count)
                # predict probabilities using this cdc
                cdc_probabilities = cdc.predict(testing_set.get_features(), True)
                # deduct the predictions of this cdc
                cdc_predicitons = cdc_probabilities > 0.5
                # calculate the mask to refine the testing set
                mask = (cdc_predicitons == testing_set.get_pseudo_labels())
                # If margin is specified and there are disagreements, check if the probabilities are significatly different
                if allow_margin and not np.all(mask):
                    # convert to disagreement mask
                    disagree_mask = ~mask
                    # calculate the difference between cdc probs and bm probs
                    prob_diff = np.abs(testing_set.get_pseudo_probabilities() - cdc_probabilities)
                    # in the disagreement mask, keep only the data point where the probability difference is greater than the margin, only for disagreed on points
                    refine_mask = (prob_diff > margin) & disagree_mask
                    # update the mask according to the refine_mask array
                    mask[refine_mask] = True
                
                # refine the testing set using the mask                
                updated_count = testing_set.refine(mask)

                # log the results for this model
                record.update(val_data_x=validation_data.get_features(), val_data_y=validation_data.get_true_labels(),
                              sample_size=updated_count, 
                              model=cdc, model_id=model_id)
                
                # break if no more data
                if updated_count == 0:
                    break

                if stopper.update(updated_count):
                    # print(f'Early stopping: Converged after {i} models')
                    break

        record.freeze()
        return record


