import pandas as pd
from sklearn import metrics
from sklearn import preprocessing


import argparse
import config 
import models_dispatcher 

import itertools



def feature_engineering(df,cat_cols):

    combi = list(itertools.combinations(cat_cols,2))
    for c1,c2 in combi:
        df.loc[:,c1+"_"+c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    
    return df


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

    df = feature_engineering(df,features)   


    features = [feature for feature in df.columns if feature not in ("income","kfold")and feature not in num_cols]

    lbl = preprocessing.LabelEncoder()

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")
        lbl.fit(df[col])
        df[col] = lbl.transform(df[col])

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [feature for feature in df.columns if feature not in ("income","kfold")]

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    
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

