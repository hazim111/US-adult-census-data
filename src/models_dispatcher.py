from sklearn import linear_model
import xgboost as xgb

MODELS = {
    "lr": linear_model.LogisticRegression(),
    "xb": xgb.XGBClassifier(n_jobs=-1)
}