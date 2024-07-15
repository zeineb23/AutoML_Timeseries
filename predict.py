def predict(file_path, dates_to_predict):
  import joblib
  import pandas as pd

  loaded_model = joblib.load(file_path)

  X_test = pd.DataFrame({'ds': pd.to_datetime(dates_to_predict)})
  predictions = loaded_model.predict(X_test)
  print(predictions)
