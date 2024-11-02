import os
import sys
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import prediction_model.processing.preprocessing as pp 

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

RogetPipeline = Pipeline(
    [
        ('Gemma7B_Embeddings', pp.Gemma7B_Embeddings()),
        ('DimensionalityReduction', pp.DimensionalityReduction()),
        ('XGBoost', XGBClassifier(use_label_encoder=False, n_jobs=-1, random_state=33))
    ]
)



