import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os 

try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError :
    TargetEncoder=None
    print("Warning: category_encoders not installed . TargetEncoding will be unavailable")

def main():

    print("Loading Datasets")
    file_path='train.csv'

    if not os.path.exists(file_path):
        print(f"Error cannot find: {file_path} ")
        return 
    df=pd.read_csv(file_path)
    # Remove duplicate header rows and convert columns to numeric
    df = df[pd.to_numeric(df['ID'], errors='coerce').notnull()].reset_index(drop=True)
    df = df.apply(pd.to_numeric)
    print(f"Dataset Loaded and Cleaned. Rows:{df.shape[0]}, Features:{df.shape[1]}\n")

    print("HANDLING MISSING DATA")
    print("Artificially Deleting some 'Hits'(H) data to demonstrate ")

    df.loc[df.sample(frac=0.25, random_state=42).index, 'H'] = np.nan

    imputer=SimpleImputer(strategy='median')
    df['H']=imputer.fit_transform(df[['H']])

    print(f"Imputation complete. 'Hits'(H) now has {df['H'].isnull().sum()} null values.\n")
    
    print("Evaluating the skewness of the Runs(R) distribution...")

    df['LogRuns']=np.log1p(df['R'])
    print(f" Log Transformation applied. New skewness:{df['LogRuns'].skew():.2f} (closer to 0 is perfectly balanced).\n")

    df['Team_ID']=['Team_'+str(np.random.randint(1,150)) for _ in range(len(df))]
    if TargetEncoder is not None:
        print("Applying Target Encoding ")
        encoder=TargetEncoder()
        df['Team_ID_Encoded'] = encoder.fit_transform(df['Team_ID'],df['W'])

        print("Target Encoding applied.\n")
    else:
        print("Category Encoders not installed.\n")

    features_to_test =['R', 'HR','SO','SB']

    x_features=df[features_to_test].fillna(0)
    y_target=df['W']

    selector=SelectKBest(score_func=mutual_info_regression,k=2)
    selector.fit(x_features, y_target)

    winning_features=selector.get_support()
    best_features=x_features.columns[winning_features].tolist()

    print(best_features)

    #splitting data
    X = df[best_features]
    Y = df["W"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print(f"Training data size: {X_train.shape}")
    print(f"Testing data size: {X_test.shape}")

    #training model

    model=LinearRegression()
    model.fit(X_train,Y_train)

    predictions=model.predict(X_test)
    print(predictions)


    #comparing model prediction to the actual answer

    actual_wins=Y_test.head(3).values
    predicted_wins=predictions[:3]

    for i in range(3):
        predicted=round(predicted_wins[i])
        actual=actual_wins[i]
        difference=abs(actual-predicted)
        
        print(f"Model Gussed:{predicted}")
        print(f"real answer:{actual}")
        print(f"differences:{difference}")

if __name__=="__main__":
    main()