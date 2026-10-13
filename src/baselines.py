import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_ROOT, IMAGE_DIR
from .data_utils import read_dicom, preprocess_image


def make_features(df, max_n=None, size=32):
    if max_n: df = df.iloc[:max_n]
    X,y=[],[]
    for _,r in df.iterrows():
        path=IMAGE_DIR/f"{r.patientId}.dcm"
        x=preprocess_image(read_dicom(path), size)
        X.append(x.ravel()); y.append(int(r.Target))
    return np.asarray(X), np.asarray(y)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--max-train",type=int,default=3000)
    p.add_argument("--max-test",type=int,default=1000)
    a=p.parse_args()
    tr=pd.read_pickle(PROCESSED_ROOT/"train.pkl")
    te=pd.read_pickle(PROCESSED_ROOT/"test.pkl")
    Xtr,ytr=make_features(tr,a.max_train); Xte,yte=make_features(te,a.max_test)
    models={
        "Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
        "Linear SVM":LinearSVC(),
        "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
    }
    rows=[]
    for name,m in models.items():
        m.fit(Xtr,ytr); pred=m.predict(Xte)
        acc=accuracy_score(yte,pred); rows.append([name,acc]); print(name,acc)
    pd.DataFrame(rows,columns=["method","accuracy"]).to_csv("results/tables/classical_baselines.csv",index=False)


if __name__=="__main__": main()
