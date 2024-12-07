heating_load_prediction/
├── data/
│   ├── historical_data.csv
│   ├── processed_data/
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   ├── raw_data/
│       ├── weather_data.csv
│       ├── heating_supply_data.csv
│       ├── wind_direction_data.csv
│       ├── engineering_indices.csv
│       ├── sunshine_duration.csv
├── models/
│   ├── lstm_model.py
│   ├── transform_model.py
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── evaluation.ipynb
├── results/
│   ├── lstm_results.csv
│   ├── transform_results.csv
│   ├── evaluation_metrics.csv
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── evaluation_metrics.py
├── README.md
├── requirements.txt
Raw Data -> Data Cleaning -> Feature Engineering -> Normalization/Scaling -> Train/Test Split
+----------------+    +----------------+    +-----------------+    +-----------------+    +-----------------+
| Raw Data Files | -> | Data Cleaning  | -> | Feature Engineering| -> | Normalization/  | -> | Train/Test    |
+----------------+    +----------------+    +-----------------+    | Scaling           |    | Split         |
                                                                    +-----------------+    +-----------------+

Input Layer -> LSTM Layer(s) -> Dense Layer -> Output Layer
+----------------+    +----------------+    +----------------+    +----------------+    +----------------+
| Input Layer    | -> | LSTM Layer(s)  | -> | Dense Layer    | -> | Output Layer   |
+----------------+    +----------------+    +----------------+    +----------------+

Preprocessed Data -> Machine Learning Algorithm -> Predictions -> Evaluation
+-----------------+    +--------------------+    +-------------+    +-------------+
| Preprocessed    | -> | Machine Learning   | -> | Predictions | -> | Evaluation  |
| Data            |    | Algorithm          |    +-------------+    +-------------+
+-----------------+    +--------------------+
