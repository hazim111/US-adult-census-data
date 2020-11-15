import pandas as pd
from sklearn import metrics
from sklearn import preprocessing


import argparse
import config 
import models_dispatcher 

def run(fold,model):

    df = pd.read_csv(config.training_data_with_folds)

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    #numerical_cols

    num_cols = [
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]

    
    # doesnt work, dont know why :(
    
    #target_mapping = {
    #    "<=50k" : 0,
    #    ">50k" : 1}

    #df["income"] = df.income.map(target_mapping)    

    #therefore we will use this :)


    df.loc[:, 'income'][df['income']== '<=50K'] = 0
    df.loc[:, 'income'][df['income']== '>50K'] = 1
    df['income'] = df.income.astype(int)

    features = [feature for feature in df.columns if feature not in ("income","kfold")and feature not in num_cols]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()
    
    # fit ohe on training + validation features

    full_data = pd.concat([df_train[features], df_valid[features]],axis=0)

    ohe.fit(full_data[features])
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    
    # modelling

    clf = models_dispatcher.MODELS[model]

    clf.fit(x_train,df_train.income.values)

    valid_preds = clf.predict_proba(x_valid)[:,1]

    # scoring
    auc = metrics.roc_auc_score(df_valid.income.values,valid_preds)
    print(f"FOLD={fold}, Accuracy = {auc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold",type=int)


    parser.add_argument("--model",type=str)

    args = parser.parse_args()

    run(fold= args.fold, model=args.model)

