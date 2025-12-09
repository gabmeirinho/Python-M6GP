import pandas

from m6gp.M6GP import M6GP
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score

import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="From version 0.21, test_size will always complement",
                        module="sklearn")

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M3GP
#
# Copyright ©2019-2025 J. E. Batista
#

runs = 20
runs_scores = []

for run in range(runs):
    print(f"Run {run+1}/{runs}")

    filename= "heart.csv"

    # Open the dataset
    ds = pandas.read_csv("datasets/"+filename)
    class_header = ds.columns[-1]

    # Split the dataset
    Tr_X, Te_X, Tr_Y, Te_Y = train_test_split(ds.drop(columns=[class_header]), ds[class_header], 
    		train_size=0.7, random_state = 42, stratify = ds[class_header])

    # Train a model
    model = M6GP(fitnesses=["2FOLD","Size"], population_size=100, max_time=600)
    model.fit(Tr_X, Tr_Y)

    # Predict test results
    pred = model.predict(Te_X)

    # Obtain test accuracy
    # print( accuracy_score(pred, Te_Y) )
    print(f1_score(pred, Te_Y, average='weighted') )

    best = model.getBestIndividual()
    print("Best individual: ", best)
    print(f"Dims: {best.getNumberOfDimensions()}, Size: {best.getSize()}")
    runs_scores.append(f1_score(pred, Te_Y, average='weighted'))

print(f"Average F1-score over {runs} runs: {sum(runs_scores)/len(runs_scores)}")

