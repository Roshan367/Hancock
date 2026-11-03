# ===================================================================================================================
# Imports
# ====================================================================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning.custom_preprocessor import HancockTabularPreprocessor
from multimodal_machine_learning.model.attention import AttentionMLPModel
from multimodal_machine_learning.predictor.abstract import (
    AbstractHancockPredictorShap,
    AbstractHancockPredictorLime,
)
from multimodal_machine_learning.model.intermediate_fusion import (
    build_intermediate_fusion_model,
)
from multimodal_machine_learning.predictor.abstract import (
    AbstractHancockPredictor,
)


# ===================================================================================================================
# Base Adjuvant Treatment Predictor using best model and data
# ====================================================================================================================
class AdjuvantTreatmentPredictor(AbstractHancockPredictorShap):
    """Class for predicting adjuvant treatments. Here the merged tabular
    data is used and a random forest classifier is trained on the data.
    The feature importance is visualized with shapley values.
    """

    def __init__(
        self, save_flag: bool = False, plot_flag: bool = False, random_state: int = 42
    ):
        """
        Initializes the AdjuvantTreatmentPredictor. It can be used to
        perform adjuvant treatment prediction with the merged tabular data.

        Args:
            save_flag (bool, optional): If this is set to True the generated
            plots will be saved. Defaults to False.

            plot_flag (bool, optional): If this is set to True the generated
            plots will be shown to the user. Defaults to False.

            random_state (int, optional): To make sure that the outputs are
            reproducible we initialize all processes that involve randomness
            with the random_state. Defaults to 42.
        """
        super().__init__(
            save_flag=save_flag,
            plot_flag=plot_flag,
            random_state=random_state,
            predictor_type="adjuvant_treatment_prediction",
        )

    # ===============================================================================================================
    # Abstract Method Implementations
    # ===============================================================================================================
    def _prepare_data(self) -> None:
        """Prepares the self._data property for the _get_df_train and _get_df_test
        methods. For this predictor the StructuralAggregated data is used.
        """
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tabular_merged_feature,
            data_dir=self.args.features_dir,
            data_dir_flag=True,
        )
        self._data = data_reader.return_data()

    def _prepare_data_split(self) -> None:
        """
        Prepares the self._data_split property for the _get_df_train and
        _get_df_test methods. For this predictor the treatment outcome data
        split is used.
        """
        data_split_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.data_split_treatment_outcome,
            data_dir=self.args.data_split_dir
            / self._default_names.data_split_treatment_outcome,
            data_dir_flag=True,
        )
        self._data_split = data_split_reader.return_data()
        self._data_split["target"] = self._data_split["adjuvant_treatment"].apply(
            lambda x: 0 if x == "none" else 1
        )

    def _get_df_train(self) -> pd.DataFrame:
        """
        Creates the training data for the predictor from the data using the
        data split and returns it as a pandas DataFrame.

        Returns:
            pd.DataFrame: The training data for the predictor.
        """
        target_train = self._data_split[self._data_split.dataset == "training"][
            ["patient_id", "target"]
        ]
        target_train = target_train.merge(self._data, on="patient_id", how="inner")
        target_train = target_train[
            ["target"] + target_train.columns.drop("target").to_list()
        ]
        return target_train

    def _get_df_test(self) -> pd.DataFrame:
        """
        Creates the test data for the predictor from the data using the
        data split and returns it as a pandas DataFrame.

        Returns:
            pd.DataFrame: The test data for the predictor.
        """
        target_test = self._data_split[self._data_split.dataset == "test"][
            ["patient_id", "target"]
        ]
        return target_test.merge(self._data, on="patient_id", how="inner")

    def _create_new_model(self) -> RandomForestClassifier:
        """
        Create a new model that is used for the prediction class.
        In this implementation a random forest classifier is used. The features
        can be explained using the shape library.

        Returns:
            RandomForestClassifier: The new model that is used for the prediction.
        """
        model = RandomForestClassifier(
            n_estimators=600,
            min_samples_split=5,
            min_samples_leaf=1,
            max_leaf_nodes=1000,
            max_features="sqrt",
            max_depth=50,
            criterion="log_loss",
            random_state=np.random.RandomState(self._random_number),
        )
        return model


# ===================================================================================================================
# Abstract Class for Neural Network Adjuvant Treatment Prediction
# ====================================================================================================================
class AbstractNeuralNetworkAdjuvantTreatmentPredictor(AbstractHancockPredictorLime):
    """Class for predicting adjuvant treatments with a neural network.
    This is an abstract class and will throw a NotImplementedError if it is
    initialized directly.
    Feature importance is visualized with LIME in the trainings process.

    To implement:
        - _prepare_data: Data getting process.

    Methods:
        cross_validate: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the adjuvant treatment for the given data.

    Properties:
        df_train: The training data.
        df_test: The test data.
        model: The model that is used for training and prediction
    """

    def __init__(
        self, save_flag: bool = False, plot_flag: bool = False, random_state: int = 42
    ):
        """
        Initializes the NeuralNetworkAdjuvantTreatmentPredictor. It can be used to
        perform adjuvant treatment prediction with the merged tabular and tma data.

        Args:
            save_flag (bool, optional): If this is set to True the generated
            plots will be saved. Defaults to False.

            plot_flag (bool, optional): If this is set to True the generated
            plots will be shown to the user. Defaults to False.

            random_state (int, optional): To make sure that the outputs are
            reproducible we initialize all processes that involve randomness
            with the random_state. Defaults to 42.
        """
        super().__init__(
            save_flag=save_flag,
            plot_flag=plot_flag,
            random_state=random_state,
            predictor_type="adjuvant_treatment_prediction",
        )
        [[x_train, _, _, _], _] = self.prepare_data_for_model(
            self.df_train, self.df_test
        )
        self._input_dim = x_train.shape[1]

    def _prepare_data_split(self) -> None:
        """
        Prepares the self._data_split property for the _get_df_train and
        _get_df_test methods. For this predictor the treatment outcome data
        split is used.
        """
        data_split_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.data_split_treatment_outcome,
            data_dir=self.args.data_split_dir
            / self._default_names.data_split_treatment_outcome,
            data_dir_flag=True,
        )
        self._data_split = data_split_reader.return_data()
        self._data_split["target"] = self._data_split["adjuvant_treatment"].apply(
            lambda x: 0 if x == "none" else 1
        )

    def _get_df_train(self) -> pd.DataFrame:
        target_train = self._data_split[self._data_split.dataset == "training"][
            ["patient_id", "target"]
        ]
        target_train = target_train.merge(self._data, on="patient_id", how="inner")
        target_train = target_train[
            ["target"] + target_train.columns.drop("target").to_list()
        ]
        return target_train

    def _get_df_test(self) -> pd.DataFrame:
        target_test = self._data_split[self._data_split.dataset == "test"][
            ["patient_id", "target"]
        ]

        merged = target_test.merge(self._data, on="patient_id", how="inner")

        return merged

    def _create_new_model(self) -> tf.keras.Model:
        """
        Create a multi layer perceptron with a single attention layer upfront
        for the prediction task. The model is compiled with the Adam optimizer.

        Returns:
            tf.keras.Model: The new model that is used for the prediction.
        """
        learning_rate = 0.001
        model = AttentionMLPModel(self._input_dim)
        model = model.build_model(self._input_dim)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model


# ===================================================================================================================
# Intermediate Fusion Adjuvant Treatment Predictor
# ===================================================================================================================



class IntermediateFusionAdjuvantTreatmentPredictor(
    AbstractNeuralNetworkAdjuvantTreatmentPredictor
):
    """Adjuvant treatment prediction using intermediate fusion."""

    TMA_FEATURE_DIM = 4096

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tma_tabular_merged_feature,
            data_dir=self.args.features_dir,
            data_dir_flag=True
        )
        self._data = data_reader.return_data()
        
    def _create_new_model(self) -> tf.keras.Model:
        """Create the intermediate-fusion neural network."""

        tabular_input_dim = self._input_dim - self.TMA_FEATURE_DIM

        if tabular_input_dim <= 0:
            raise ValueError(
                f"Invalid tabular dimension: {tabular_input_dim}. "
                f"Total features: {self._input_dim}"
            )

        model = build_intermediate_fusion_model(
            tma_input_dim=self.TMA_FEATURE_DIM,
            tabular_input_dim=tabular_input_dim,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(
                    threshold=0.0,
                    name="accuracy",
                )
            ],
        )

        return model

    def _split_features(self, x):
        """Separate TMA embeddings from tabular features."""

        tma = x[:, -self.TMA_FEATURE_DIM:]
        tabular = x[:, :-self.TMA_FEATURE_DIM]

        return [tma, tabular]

    def train(
        self,
        df_train: pd.DataFrame = None,
        df_other: pd.DataFrame = None,
        plot_name: str = 'intermediate_fusion',
        model_reset: bool = True,
        **kwargs
    ) -> list:

        df_train, df_other = self._setup_training(
            df_train, df_other, model_reset
        )

        print("Training class distribution:")
        print(df_train["target"].value_counts())

        print("\nValidation class distribution:")
        print(df_other["target"].value_counts())

        [[x_train, y_train, x_other, y_other], features] = (
            self.prepare_data_for_model(df_train, df_other)
        )

        # Separate the TMA and tabular inputs
        x_train = self._split_features(x_train)
        x_other = self._split_features(x_other)

        # Convert one-hot labels to binary labels
        y_train = y_train[:, 1]
        y_other = y_other[:, 1]

        self.model.fit(x_train, y_train)

        # Predict probabilities
        y_pred = self.predict(x_other)

        # Calculate evaluation metrics
        fpr, tpr, auc_score = self._training_calculate_metrics(
            y_other, y_pred
        )

        if self._plotter.plot_flag or self._plotter.save_flag:
            self._plot_train(df_other, y_pred, plot_name)

        return [
            [fpr, tpr, auc_score],
            [x_train, y_train, x_other, y_other, y_pred],
            features
        ]

    def predict(self, data):
        """Return predicted probabilities."""

        logits = self.model.predict(data, verbose=0)
        return tf.math.sigmoid(logits).numpy().reshape(-1)

    def cross_validate(self, *args, **kwargs):
        """Use HANCOCK's original cross-validation implementation."""

        return AbstractHancockPredictor.cross_validate(
            self,
            *args,
            **kwargs,
        )


# ===================================================================================================================
# Usable implementation with Data and Model - Naming convention: [Data][Model]AdjuvantTreatmentPredictor
# ====================================================================================================================
class TabularMergedAttentionMlpAdjuvantTreatmentPredictor(
    AbstractNeuralNetworkAdjuvantTreatmentPredictor
):
    """Class for predicting adjuvant treatments with a multi layer perceptron
    where the features are fused with a simple attention mechanism.
    This implementation uses the merged tabular and tma feature data.

    Methods:
        cross_validate: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the adjuvant treatment for the given data.

    Properties:
        df_train: The training data.
        df_test: The test data.
        model: The model that is used for training and prediction
    """

    def _prepare_data(self) -> None:
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tabular_merged_feature,
            data_dir=self.args.features_dir,
            data_dir_flag=True,
        )
        self._data = data_reader.return_data()


class TmaTabularMergedWithoutPathologicalDataTmaCellDensityAdjuvantTreatmentPredictor(
    AbstractNeuralNetworkAdjuvantTreatmentPredictor
):
    """Class for predicting adjuvant treatments with a multi layer perceptron
    where the features are fused with a simple attention mechanism.
    This implementation uses the merged tabular and tma feature data, but
    does not use the pathological features as well as the TMA vector
    features for CD3 and CD8.

    Methods:
        cross_validate: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the adjuvant treatment for the given data.

    Properties:
        df_train: The training data.
        df_test: The test data.
        model: The model that is used for training and prediction
    """

    def _prepare_data(self) -> None:
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tma_tabular_without_patho_tma_cell_density_feature,
            data_dir=self.args.features_dir,
            data_dir_flag=True,
        )
        self._data = data_reader.return_data()


class TmaAttentionMlpAdjuvantTreatmentPredictor(
    AbstractNeuralNetworkAdjuvantTreatmentPredictor
):
    """Class for predicting adjuvant treatments with a multi layer perceptron
    where the features are fused with a simple attention mechanism.
    This implementation only uses the TMA vector features.

    Methods:
        cross_validate: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the adjuvant treatment for the given data.

    Properties:
        df_train: The training data.
        df_test: The test data.
        model: The model that is used for training and prediction
    """

    def _prepare_data(self) -> None:
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tma_merged_feature,
            data_dir=self.args.features_dir,
            data_dir_flag=True,
        )
        self._data = data_reader.return_data()


class TabularMergedAdjuvantTreatmentPredictor(AdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the merged tabular
    data is used for the prediction.
    """

    def _prepare_data(self) -> None:
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tabular_merged_feature,
            data_dir=self.args.features_dir,
            data_dir_flag=True,
        )
        self._data = data_reader.return_data()

    def _prepare_data_split(self) -> None:
        """Prepares the self._data_split property for the _get_df_train and
        _get_df_test methods. For this predictor the treatment outcome data
        split is used.
        """
        data_split_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.data_split_treatment_outcome,
            data_dir=self.args.data_split_dir
            / self._default_names.data_split_treatment_outcome,
            data_dir_flag=True,
        )
        self._data_split = data_split_reader.return_data()
        self._data_split["target"] = self._data_split["adjuvant_treatment"].apply(
            lambda x: 0 if x == "none" else 1
        )

    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=600,
            min_samples_split=5,
            min_samples_leaf=1,
            max_leaf_nodes=1000,
            max_features="sqrt",
            max_depth=50,
            criterion="log_loss",
            random_state=np.random.RandomState(self._random_number),
        )
        return model


class ClinicalAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the clinical tabular
    data is used.
    """

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.clinical_feature
        )
        self._data = data_reader.return_data()


class PathologicalAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the pathological tabular
    data is used.
    """

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.patho_feature
        )
        self._data = data_reader.return_data()


class BloodAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the blood tabular
    data is used.
    """

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.blood_feature
        )
        self._data = data_reader.return_data()


class TMACellDensityAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the TMA cell density tabular
    data is used.
    """

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tma_cell_density_feature
        )
        self._data = data_reader.return_data()


class ICDCodesAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the ICD codes are
    is used."""

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.icd_codes_feature
        )
        self._data = data_reader.return_data()


# ===================================================================================================================
# Functions for running the predictors
# ====================================================================================================================
def adjuvant_treatment_prediction_tma_vector_attention(save_flag=True, plot_flag=True):
    predictors = [
        TabularMergedAttentionMlpAdjuvantTreatmentPredictor(
            save_flag=save_flag, plot_flag=plot_flag
        ),
        TmaAttentionMlpAdjuvantTreatmentPredictor(
            save_flag=save_flag, plot_flag=plot_flag
        ),
        TmaTabularMergedWithoutPathologicalDataTmaCellDensityAdjuvantTreatmentPredictor(
            save_flag=save_flag, plot_flag=plot_flag
        ),
    ]
    plot_base_name = "attention_mlp_"
    predictor_names = [
        "tma_tabular_merged",
        "tma",
        "tma_tabular_merged_without_patho_tma_cell_density",
    ]
    plot_lime = [True, False, False]

    for i, (predictor, predictor_name) in enumerate(zip(predictors, predictor_names)):
        print(f"Running training for {predictor_name} data ...")
        _ = predictor.train(
            plot_name=plot_base_name + predictor_name + "_train", lime_flag=plot_lime[i]
        )
        print(f"Running k-fold cross-validation for {predictor_name} data ...")
        _ = predictor.cross_validate(
            plot_name=plot_base_name + predictor_name + "_cross_val"
        )


def adjuvant_treatment_prediction_tabular_only(save_flag=True, plot_flag=True):
    multi_predictor = TabularMergedAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag
    )
    clinical_predictor = ClinicalAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag
    )
    patho_predictor = PathologicalAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag
    )
    blood_predictor = BloodAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag
    )
    tma_cell_density_predictor = TMACellDensityAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag
    )
    icd_predictor = ICDCodesAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag
    )
    cross_validation_splits = 10

    data_types = [
        "multimodal",
        "clinical",
        "pathological",
        "blood",
        "cell density",
        "text",
    ]
    predictors = [
        multi_predictor,
        clinical_predictor,
        patho_predictor,
        blood_predictor,
        tma_cell_density_predictor,
        icd_predictor,
    ]

    for data_type, predictor in zip(data_types, predictors):
        print(f"Running k-fold cross-validation for {data_type} data ...")
        _ = predictor.cross_validate(
            n_splits=cross_validation_splits, plot_name="adjuvant_therapy_" + data_type
        )

    # Train classifier once on multimodal data, show survival curves and bar plot
    print("Training and testing the final multimodal model...")
    multi_predictor.train(
        plot_name="adjuvant_therapy_multimodal",
        df_train=multi_predictor.df_train,
        df_other=multi_predictor.df_test,
    )


def adjuvant_treatment_prediction_intermediate_fusion(
    save_flag=True,
    plot_flag=True,
):
    predictor = IntermediateFusionAdjuvantTreatmentPredictor(
        save_flag=save_flag,
        plot_flag=plot_flag,
    )

    print("Running intermediate-fusion training...")
    predictor.train(
        plot_name="intermediate_fusion_train",
        df_train=predictor.df_train,
        df_other=predictor.df_test,
    )

    print("Running intermediate-fusion cross-validation...")
    predictor.cross_validate(
        n_splits=10,
        plot_name="intermediate_fusion_cross_val",
    )