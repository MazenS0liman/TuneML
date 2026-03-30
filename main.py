import mlflow


mlflow.set_experiment("Youtube Tutorial")

with mlflow.start_run(run_name="Logging Demo"):
    mlflow.log_param()
    
    mlflow.log_param("param1", 5)
    
    
    
    
    
