"""Counterfactual Recurrent Network. Treatment effects model."""

import os
import pickle

import numpy as np
from base import BaseEstimator, PredictorMixin
from treatments.CRN.CRN_Base import CRN_Base
from treatments.CRN.data_utils import (
    data_preprocess,
    data_preprocess_counterfactuals,
    process_seq_data,
)


class CRN_Model(BaseEstimator, PredictorMixin):  # pylint: disable=abstract-method
    def __init__(self, hyperparams_encoder=None, hyperparams_decoder=None, task=None, static_mode=None, time_mode=None):
        """
        Initialize the Counterfactual Recurrent Network (CRN).

        Args:
            - hyperparams_encoder: dictionary with the hyperparameters specifying the architecture of the CRN encoder model.
            - hyperparams_decoder: dictionary with the hyperparameters specifying the architecture of the CRN decoder model.
            - task: 'classification' or 'regression'
            - static_mode: 'concatenate' or None
            - time_mode: 'concatenate' or None
        """
        super().__init__(task)

        self.task = task
        self.static_mode = static_mode
        self.time_mode = time_mode

        self.hyperparams_encoder = hyperparams_encoder
        self.hyperparams_decoder = hyperparams_decoder

    def fit(
        self, dataset, projection_horizon=None, fold=0, train_split="train", val_split="val"
    ):  # pylint: disable=arguments-differ
        """Fit the treatment effects encoder model model.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - projection_horizon: number of future timesteps to use for training decoder model.
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
        """
        dataset_crn_train = data_preprocess(dataset, fold, train_split, self.static_mode, self.time_mode)
        dataset_crn_val = data_preprocess(dataset, fold, val_split, self.static_mode, self.time_mode)

        num_outputs = dataset_crn_train["outputs"].shape[-1]
        max_sequence_length = dataset_crn_train["current_covariates"].shape[1]
        num_covariates = dataset_crn_train["current_covariates"].shape[-1]

        self.hyperparams_decoder["rnn_hidden_units"] = self.hyperparams_encoder["br_size"]

        self.encoder_params = {
            "num_treatments": 2,
            "num_covariates": num_covariates,
            "num_outputs": num_outputs,
            "max_sequence_length": max_sequence_length,
        }

        self.encoder_model = CRN_Base(self.hyperparams_encoder, self.encoder_params, task=self.task)
        self.encoder_model.train(dataset_crn_train, dataset_crn_val)

        if projection_horizon is not None:
            training_br_states = self.encoder_model.get_balancing_reps(dataset_crn_train)
            validation_br_states = self.encoder_model.get_balancing_reps(dataset_crn_val)

            training_seq_processed = process_seq_data(dataset_crn_train, training_br_states, projection_horizon)
            validation_seq_processed = process_seq_data(dataset_crn_val, validation_br_states, projection_horizon)

            num_outputs = training_seq_processed["outputs"].shape[-1]
            num_covariates = training_seq_processed["current_covariates"].shape[-1]

            self.decoder_params = {
                "num_treatments": 2,
                "num_covariates": num_covariates,
                "num_outputs": num_outputs,
                "max_sequence_length": projection_horizon,
            }

            self.decoder_model = CRN_Base(
                self.hyperparams_decoder, self.decoder_params, b_train_decoder=True, task=self.task
            )
            self.decoder_model.train(training_seq_processed, validation_seq_processed)

    def predict(self, dataset, fold=0, test_split="test"):  # pylint: disable=arguments-differ
        """Return the one-step-ahead predicted outcomes on the test set. These are one-step-ahead predictions.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Test fold
            - test_split: testing set splitting parameter

        Returns:
            - test_y_hat: predictions on testing set
        """
        dataset_crn_test = data_preprocess(dataset, fold, test_split, self.static_mode, self.time_mode)
        test_y_hat = self.encoder_model.get_predictions(dataset_crn_test)
        return test_y_hat

    def predict_counterfactual_trajectories(
        self, dataset, patient_id, timestep, treatment_options, fold=0, test_split="test"
    ):
        """Return the counterfactual trajectories for a patient and for the specified future treatment options.

        Args:
            - dataset: dataset with test patients
            - patient_id: patient id of patient for which the counterfactuals are computed
            - timestep: timestep in the patient trajectory where counterfactuals are predicted
            - treatment_options: treatment options for computing the counterfactual trajectories; the length of the
                sequence of treatment options needs to be projection_horizon + 1 where projection_horizon is the number of
                future timesteps used for training decoder model.
            - fold: test fold
            - test_split: testing set splitting parameter

        Returns:
            - history: history of previous outputs for the patient.
            - counterfactual_trajectories: trajectories of counterfactual predictions for the specified future treatments
                in the treatment_options
        """
        history, encoder_output, dataset_crn_decoder = data_preprocess_counterfactuals(
            encoder_model=self.encoder_model,
            dataset=dataset,
            patient_id=patient_id,
            timestep=timestep,
            treatment_options=treatment_options,
            fold=fold,
            split=test_split,
            static_mode=self.static_mode,
            time_mode=self.time_mode,
        )
        decoder_outputs = self.decoder_model.get_autoregressive_sequence_predictions(dataset_crn_decoder)
        counterfactual_trajectories = np.concatenate([encoder_output, decoder_outputs], axis=1)

        return history, counterfactual_trajectories

    def save_model(self, model_dir, model_name):  # pylint: disable=arguments-differ
        """Save the model to model_dir using the model_name.

        Args:
            - model_dir: directory where to save the model
            - model_name: name of saved model
        """
        encoder_model_name = "encoder_" + model_name
        self.encoder_model.save_network(model_dir, encoder_model_name)
        pickle.dump(self.encoder_params, open(os.path.join(model_dir, "encoder_params_" + model_name + ".pkl"), "wb"))
        pickle.dump(
            self.hyperparams_encoder, open(os.path.join(model_dir, "hyperparams_encoder_" + model_name + ".pkl"), "wb")
        )

        decoder_model_name = "decoder_" + model_name
        self.decoder_model.save_network(model_dir, decoder_model_name)
        pickle.dump(self.decoder_params, open(os.path.join(model_dir, "decoder_params_" + model_name + ".pkl"), "wb"))
        pickle.dump(
            self.hyperparams_decoder, open(os.path.join(model_dir, "hyperparams_decoder_" + model_name + ".pkl"), "wb")
        )

    def load_model(self, model_dir, model_name):  # pylint: disable=arguments-differ
        """
        Load and return the model from model_path

        Args:
            - model_path:  dictionary containing model_dir (directory where to save the model) and model_name for the
                        the saved encoder and decoder models
        """
        encoder_params = pickle.load(open(os.path.join(model_dir, "encoder_params_" + model_name + ".pkl"), "rb"))
        encoder_hyperparams = pickle.load(
            open(os.path.join(model_dir, "hyperparams_encoder_" + model_name + ".pkl"), "rb")
        )
        encoder_model_name = "encoder_" + model_name

        encoder_model = CRN_Base(encoder_hyperparams, encoder_params, task=self.task)
        encoder_model.load_model(model_name=encoder_model_name, model_folder=model_dir)

        decoder_params = pickle.load(open(os.path.join(model_dir, "decoder_params_" + model_name + ".pkl"), "rb"))
        decoder_hyperparams = pickle.load(
            open(os.path.join(model_dir, "hyperparams_decoder_" + model_name + ".pkl"), "rb")
        )
        decoder_model_name = "decoder_" + model_name
        decoder_model = CRN_Base(decoder_hyperparams, decoder_params, b_train_decoder=True, task=self.task)
        decoder_model.load_model(model_name=decoder_model_name, model_folder=model_dir)

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
