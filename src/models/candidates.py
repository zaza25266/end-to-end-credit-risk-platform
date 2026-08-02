from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from src.utils.config import load_params

def get_candidate_models():
    """
    Initializes CatBoost, XGBoost, LightGBM, and the Soft Voting Ensemble
    using hyperparameter configurations loaded directly from params.yaml.
    """
    params = load_params()
    models_cfg = params["models"]
    ensemble_cfg = params["ensemble"]

    # 1. CatBoost Classifier
    cb_params = models_cfg["catboost"]["params"]
    cb_fixed = models_cfg["catboost"]["fixed_params"]
    catboost_model = CatBoostClassifier(**cb_params, **cb_fixed)

    # 2. XGBoost Classifier
    xgb_params = models_cfg["xgboost"]["params"]
    xgb_fixed = models_cfg["xgboost"]["fixed_params"]
    xgboost_model = XGBClassifier(**xgb_params, **xgb_fixed)

    # 3. LightGBM Classifier
    lgb_params = models_cfg["lightgbm"]["params"]
    lgb_fixed = models_cfg["lightgbm"]["fixed_params"]
    lightgbm_model = LGBMClassifier(**lgb_params, **lgb_fixed)

    # 4. Soft Voting Ensemble (CatBoost + LightGBM as specified in params.yaml)
    weights = [
        ensemble_cfg["weights"]["catboost"],
        ensemble_cfg["weights"]["lightgbm"]
    ]
    
    ensemble_model = VotingClassifier(
        estimators=[
            ("catboost", catboost_model),
            ("lightgbm", lightgbm_model)
        ],
        voting=ensemble_cfg["voting"],
        weights=weights
    )

    return {
        "catboost": catboost_model,
        "xgboost": xgboost_model,
        "lightgbm": lightgbm_model,
        "ensemble": ensemble_model
    }