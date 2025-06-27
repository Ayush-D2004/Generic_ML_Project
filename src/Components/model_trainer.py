import os
import sys
from src.exceptions import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info(
                "Splitting training and testing arrays into input and target features"
            )
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "ElasticNet Regression": ElasticNet(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=0),
                "XGBoost Regressor": XGBRegressor(),
            }
            model_report: dict = evaluate_models(
                xtrain=X_train,
                ytrain=y_train,
                xtest=X_test,
                ytest=y_test,
                models=models,
            )

            logging.info(f"Model report: {model_report}")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(
                f"Best model found: {best_model_name} with R2 score: {best_model_score}"
            )

            param_grid = {
                "Linear Regression": {},
                "Ridge Regression": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
                "Lasso Regression": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
                "ElasticNet Regression": {
                    "alpha": [0.001, 0.01, 0.1, 1.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                },
                "KNeighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"],
                },
                "Decision Tree Regressor": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["squared_error", "friedman_mse"],
                },
                "Random Forest Regressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "bootstrap": [True, False],
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"],
                },
                "Gradient Boosting Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                },
                "CatBoost Regressor": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [3, 5, 7],
                    "l2_leaf_reg": [1, 3, 5],
                },
                "XGBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },
            }

            grid_search = GridSearchCV(
                estimator=best_model,
                param_grid=param_grid[best_model_name],
                scoring="r2",
                cv=3,
                verbose=1,
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_

            best_tuned_model = grid_search.best_estimator_

            predicted = best_tuned_model.predict(X_test)
            r2_square = best_tuned_model.score(X_test, y_test)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_tuned_model,
            )

            logging.info(
                f"Best model is {best_model_name} with parameters: {best_params} got r2_score: {r2_square}"
            )
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
