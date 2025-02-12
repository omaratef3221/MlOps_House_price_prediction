from train import train
import argparse
from data import get_cleaned_dataset
import os
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://127.0.0.1:8080") ## Track Experiment
mlflow.set_experiment("House Price Prediction") ## Set the experiment name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--remove_outliers', type=int, default=0)
    args = parser.parse_args()


    if os.path.exists(args.model_path) == False:
        os.makedirs(args.model_path)
    
    df = get_cleaned_dataset("perth-house-prices/all_perth_310121.csv", encoder_path=args.model_path, args = args)
    results, model = train(args, df)

    dataset = mlflow.data.from_pandas(df, source='perth-house-prices/all_perth_310121.csv', name="Perth Houses", targets="PRICE")
    X = df.drop(["PRICE"], axis = 1)
    signature = infer_signature(X,model.predict(X))
    
    with mlflow.start_run(run_name = args.algorithm):
        mlflow.log_param("Algorithm Name",args.algorithm)
        mlflow.log_metric("MSE", results["MSE"])
        mlflow.log_metric("RMSE", results["RMSE"])
        mlflow.log_metric("R2_Score", results["R2_Score"])
        mlflow.log_metric("MAE Percentage", results["MAE Percentage"])
        mlflow.log_artifacts(args.model_path, artifact_path=args.model_path)
        mlflow.log_input(dataset, context="training")
    
        model_info=mlflow.sklearn.log_model(
                sk_model = model,
                artifact_path=args.model_path,
                signature=signature,
                input_example=X,
                registered_model_name=args.algorithm,
            )

        