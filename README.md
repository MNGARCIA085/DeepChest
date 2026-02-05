# DeepChest





### MLFLow


#### Run server
```
...$ mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host localhost \
    --port 5000
```

#### Delete the SQLite file
```
rm mlflow.db
```

#### Delete all artifacts (by default in ./mlruns/)
```
rm -rf mlruns/
```