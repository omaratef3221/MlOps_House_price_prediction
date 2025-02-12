from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor
import numpy as np
from data import get_cleaned_dataset
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
import joblib


def train(args, data):
    Y = data["PRICE"]
    X = data.drop(["PRICE"], axis = 1)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    if args.algorithm == "random_forest":
        algo = RandomForestRegressor()
    elif args.algorithm == "linear_regression":
        algo = LinearRegression()
    elif args.algorithm == "neural_network":
        algo = MLPRegressor(max_iter=100, hidden_layer_sizes=(10, 10), batch_size=32, learning_rate= "adaptive")

    algo.fit(X_train, y_train)
    predictions = algo.predict(X_test)
    
    print("MSE: ", mean_squared_error(y_test, predictions))
    print("RMSE: ", root_mean_squared_error(y_test, predictions))
    print("R2_Score: ", r2_score(predictions, y_test))
    print("MAE Percentage: ", mean_absolute_percentage_error(y_test, predictions))

    joblib.dump(algo, args.model_path + "/model.pkl")
    
    results = {
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": root_mean_squared_error(y_test, predictions),
        "R2_Score": r2_score(predictions, y_test),
        "MAE Percentage": mean_absolute_percentage_error(y_test, predictions)
    }
    return results, algo

