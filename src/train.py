import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocess import load_and_preprocess

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess("data/BMW_sales_data_2010_2024.csv")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("BMW-Sales-Model")

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
