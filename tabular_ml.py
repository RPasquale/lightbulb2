import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder, normalize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import VotingRegressor
from sklearn.feature_selection import SelectFromModel
import optuna
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import json
import os
from datetime import datetime
import joblib
import networkx as nx
from transformers import BertTokenizer, BertModel
from scipy.stats import spearmanr

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DATASET_FOLDER = r'C:\Users\Admin\Desktop\ai-ml-web-app\backend\data'

def numpy_to_python(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, device, lr=0.001, epochs=100, batch_size=32):
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy().squeeze()
        return predictions

class TabularMLProcessor:
    def __init__(self, results_dir = 'results'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nn_params = {
            'hidden_size': [64, 128, 256],
            'lr': [1e-4, 1e-3, 1e-2],
            'batch_size': [32, 64, 128],
            'epochs': [50, 100, 200]
        }
        self.xgb_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        self.models = {
            'xgboost': None,
            'neural_network': None,
            'ensemble': None
        }
        self.preprocessor = None
        self.results_dir = results_dir
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.knowledge_graph = nx.Graph()

        # Ensure the results directory exists
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            logger.info(f"Created directory: {self.results_dir}")

        self.models_dir = os.path.join(results_dir, 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.label_encoders = {}

    def create_torch_model(self, input_size, hidden_size=128):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
        return model.to(self.device)

    def train_torch_model(self, model, X_train, y_train, X_val, y_val, lr=0.001, epochs=100, batch_size=32):
        # Ensure that y_train and y_val are NumPy arrays first
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience = 10
        counter = 0
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        # Ensure all inputs are numpy arrays
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        X_val = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        y_val = y_val.values if isinstance(y_val, pd.Series) else y_val

        # Tuning XGBoost
        def xgb_objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist',  # Use GPU for training
                'device': 'cuda',  # Specify GPU device
                'eval_metric': 'rmse',
                'importance_type': 'weight',
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }

            # Create DMatrix for XGBoost
            xgb_train = xgb.DMatrix(X_train, label=y_train)
            xgb_val = xgb.DMatrix(X_val, label=y_val)
            
            evals = [(xgb_train, 'train'), (xgb_val, 'eval')]
            
            # Train the model using GPU
            model = xgb.train(params, xgb_train, num_boost_round=300, evals=evals, early_stopping_rounds=10, verbose_eval=False)
            
            # Make predictions
            preds = model.predict(xgb_val, iteration_range=(0, model.best_iteration))
            mse = mean_squared_error(y_val, preds)
            
            return mse

        xgb_study = optuna.create_study(direction='minimize')
        xgb_study.optimize(xgb_objective, n_trials=50)
        self.models['xgboost'] = xgb.XGBRegressor(**xgb_study.best_params)
        self.models['xgboost'].fit(X_train, y_train)

        # Tuning Neural Network
        def nn_objective(trial):
            hidden_size = trial.suggest_categorical('hidden_size', self.nn_params['hidden_size'])
            lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', self.nn_params['batch_size'])
            epochs = trial.suggest_categorical('epochs', self.nn_params['epochs'])

            model = self.create_torch_model(input_size=X_train.shape[1], hidden_size=hidden_size)
            self.train_torch_model(model, X_train, y_train, X_val, y_val, lr=lr, epochs=epochs, batch_size=batch_size)
            model.eval()

            # Convert X_val to a PyTorch tensor if it's a NumPy array
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                val_pred = model(X_val_tensor).cpu().numpy()
            
            val_score = mean_squared_error(y_val, val_pred)
            return val_score

        nn_study = optuna.create_study(direction='minimize')
        nn_study.optimize(nn_objective, n_trials=50)
        best_params = nn_study.best_params

        self.models['neural_network'] = self.create_torch_model(input_size=X_train.shape[1], hidden_size=best_params['hidden_size'])
        self.train_torch_model(self.models['neural_network'], X_train, y_train, X_val, y_val, lr=best_params['lr'], epochs=best_params['epochs'], batch_size=best_params['batch_size'])
        
    def feature_optimization(self, X, y):
        # Feature importance based selection
        print("Starting feature optimization...")

        # Initial check on the data
        print(f"Input data shape before optimization: {X.shape}")
        print(f"Input data type: {type(X)}")
        print(f"Column names: {X.columns.tolist()}")

        # Initialize and fit XGBoost model for feature selection
        xgb_model = xgb.XGBRegressor(n_estimators=100)
        xgb_model.fit(X, y)

        # Feature importance output for debug
        importance = xgb_model.feature_importances_
        print(f"Feature importances: {importance}")

        # Feature selection based on importance
        selector = SelectFromModel(xgb_model, prefit=True, threshold='median')
        X_selected = selector.transform(X)

        selected_features = X.columns[selector.get_support()]
        
        print(f"Selected features: {selected_features.tolist()}")
        print(f"Optimized data shape: {X_selected.shape}")
        
        return X_selected, selected_features

    def preprocess_data(self, df, target_column):
        def safe_float_convert(x):
            if isinstance(x, str):
                x = x.strip()  # Remove leading and trailing whitespace
                logger.debug(f"Converting value: '{x}'")

                # Handle special cases for strings that should map to NaN or zero
                if x in ['', '-', ' -   ', 'N/A', 'NaN', '$-']:
                    logger.debug(f"Converting '{x}' to NaN")
                    return np.nan

                # Remove common currency symbols and thousands separators
                x = x.replace('$', '').replace(',', '').replace(' ', '')

                # Handle negative numbers represented as '$-'
                if x == '-':
                    logger.debug("Converting '-' to -0.0")
                    return -0.0

            try:
                float_value = float(x)
                logger.debug(f"Successfully converted '{x}' to {float_value}")
                return float_value
            except ValueError:
                logger.error(f"Failed to convert value: '{x}'")
                return np.nan

        logger.info(f"Starting preprocessing for target column: {target_column}")
        logger.debug(f"Original dataframe shape: {df.shape}")
        logger.debug(f"Original dataframe columns: {df.columns.tolist()}")

        # Apply safe_float_convert to all columns
        for col in df.columns:
            logger.info(f"Converting column: {col}")
            df[col] = df[col].apply(safe_float_convert)

        y = df[target_column]
        X = df.drop(target_column, axis=1)

        # Remove rows where target is NaN, inf, or extremely large
        valid_indices = ~(y.isna() | np.isinf(y) | (np.abs(y) > 1e300))
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

        logger.info(f"After removing invalid rows - X shape: {X.shape}, y shape: {y.shape}")

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        logger.info(f"Numeric features: {numeric_features.tolist()}")
        logger.info(f"Categorical features: {categorical_features.tolist()}")

        # Create preprocessing pipeline for numeric data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Create preprocessing pipeline for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine preprocessors
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ], remainder='passthrough'  # Pass through any columns not processed
        )

        # Fit the preprocessor and transform the data
        logger.info("Fitting preprocessor...")
        X_preprocessed = self.preprocessor.fit_transform(X)
        logger.info("Preprocessor fitted and data transformed.")

        # Get feature names from the preprocessor
        feature_names = []
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat' and len(features) > 0:
                cat_encoder = transformer.named_steps['onehot']
                if hasattr(cat_encoder, 'get_feature_names_out'):
                    feature_names.extend(cat_encoder.get_feature_names_out(features))

        # Log preprocessed data shape and feature names
        logger.debug(f"Preprocessed data shape: {X_preprocessed.shape}")
        logger.debug(f"Number of feature names: {len(feature_names)}")
        logger.debug(f"Feature names: {feature_names}")

        # Ensure the number of feature names matches the number of columns in X_preprocessed
        if X_preprocessed.shape[1] != len(feature_names):
            logger.warning(f"Mismatch between preprocessed data shape ({X_preprocessed.shape[1]}) and number of feature names ({len(feature_names)})")
            
            if X_preprocessed.shape[1] > len(feature_names):
                # If we have more columns than names, add generic names for the extra columns
                extra_features = [f'unknown_feature_{i}' for i in range(X_preprocessed.shape[1] - len(feature_names))]
                logger.warning(f"Adding generic names for {len(extra_features)} extra features: {extra_features}")
                feature_names.extend(extra_features)
            else:
                # If we have more names than columns, truncate the list of names
                logger.warning(f"Truncating feature names list to match data shape. Removed names: {feature_names[X_preprocessed.shape[1]:]}")
                feature_names = feature_names[:X_preprocessed.shape[1]]

        # Set feature names as a class attribute
        self.feature_names = feature_names

        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)

        logger.info("Preprocessing complete")
        return X_preprocessed_df, y

    def process(self, data, target_column, dataset_name):
        try:
            logger.info("Starting tabular ML process")
            df = pd.DataFrame(data)
            
            # Check if target column exists
            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found in the dataset")
                raise ValueError(f"Target column '{target_column}' not found in the dataset")

            logger.debug(f"Original data shape: {df.shape}")
            logger.debug(f"Original data columns: {df.columns.tolist()}")
            logger.debug(f"Target column data (first 5 rows): {df[target_column].head()}")
            logger.debug(f"Target column dtype: {df[target_column].dtype}")

            # Preprocess data
            X, y = self.preprocess_data(df, target_column)

            logger.info(f"Preprocessed data shape: {X.shape}")
            logger.debug(f"Preprocessed data columns: {X.columns.tolist()}")
            logger.debug(f"Preprocessed target data (first 5 rows): {y.head()}")

            # Check if there's enough data after preprocessing
            if len(X) < 100:
                raise ValueError("Not enough valid data for analysis. Minimum 100 rows required after preprocessing.")

            # Feature optimization
            X_optimized, selected_features = self.feature_optimization(X, y)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_optimized, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Ensure all data is in numpy array format
            X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            X_val = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
            y_val = y_val.values if isinstance(y_val, pd.Series) else y_val
            y_test = y_test.values if isinstance(y_test, pd.Series) else y_test

            print(f"Training data shape: {X_train.shape}")
            print(f"Validation data shape: {X_val.shape}")
            print(f"Test data shape: {X_test.shape}")

            # Hyperparameter Tuning
            print("Starting hyperparameter tuning...")
            self.hyperparameter_tuning(X_train, y_train, X_val, y_val)

            # Create and train the NeuralNetworkRegressor
            nn_model = self.create_torch_model(input_size=X_train.shape[1])
            nn_regressor = NeuralNetworkRegressor(
                model=nn_model,
                device=self.device,
                lr=0.001,
                epochs=100,
                batch_size=32
            )
            nn_regressor.fit(X_train, y_train)

            # Update the models dictionary
            self.models['neural_network'] = nn_regressor

            # Train Ensemble
            print("Training ensemble model...")
            self.models['ensemble'] = VotingRegressor([
                ('xgboost', self.models['xgboost']),
                ('neural_network', nn_regressor)
            ])
            self.models['ensemble'].fit(X_train, y_train)

            # Evaluate models
            results = {}
            for name, model in self.models.items():
                y_pred = model.predict(X_test)
                mse = float(mean_squared_error(y_test, y_pred)) if len(y_test) > 0 else None
                mae = float(mean_absolute_error(y_test, y_pred)) if len(y_test) > 0 else None
                r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 0 else None
                results[name] = {'mse': mse, 'mae': mae, 'r2': r2}

            # Ensure no metrics are missing
            for model_name in ['xgboost', 'neural_network', 'ensemble']:
                if model_name not in results:
                    results[model_name] = {'mse': None, 'mae': None, 'r2': None}

            # Generate embeddings
            embeddings = self.generate_tabular_embeddings(X)
            results['embeddings'] = embeddings.tolist()

            # Build knowledge graph
            feature_importances = dict(zip(X.columns, self.models['xgboost'].feature_importances_))
            self.build_knowledge_graph(X, y, feature_importances)
            results['knowledge_graph'] = nx.node_link_data(self.knowledge_graph)

            # Feature importance
            sorted_feature_importance = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            results['feature_importance'] = [(str(feature), float(importance)) for feature, importance in sorted_feature_importance[:10]]

            # Generate insights
            insights = self.generate_insights(results)

            # Store analysis results and insights
            self.analysis_results = {
                "dataset": df.to_dict(),
                "target_column": target_column,
                "model_metrics": results,
                "insights": insights
            }

            # Save the best model
            best_model_filename = self.save_best_model(results, dataset_name, target_column)
            results['best_model_filename'] = best_model_filename

            # Save the results after processing
            result_file_path = self.save_results(dataset_name=dataset_name)
            logger.info(f"Results saved to {result_file_path}")

            # Generate LLM context
            llm_context = self.generate_llm_context()

            return numpy_to_python(results), llm_context
        except Exception as e:
            logger.exception(f"Error in process method: {str(e)}")
            raise Exception(f'Error processing tabular data: {str(e)}')

    def generate_llm_context(self):
        context_lines = []
        
        # Add insights to the context
        if 'insights' in self.analysis_results:
            context_lines.append("Insights:")
            context_lines.extend(self.analysis_results['insights'])

        # Add model performance metrics to the context
        if 'model_metrics' in self.analysis_results:
            context_lines.append("\nModel Performance:")
            for model_name, metrics in self.analysis_results['model_metrics'].items():
                if isinstance(metrics, dict) and model_name != 'embeddings' and model_name != 'knowledge_graph':
                    context_lines.append(f"{model_name} - MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")

        # Add feature importance
        if 'feature_importance' in self.analysis_results['model_metrics']:
            context_lines.append("\nTop Features Affecting the Target:")
            for feature, importance in self.analysis_results['model_metrics']['feature_importance']:
                context_lines.append(f"{feature}: {importance:.4f}")

        # Add embedding information
        if 'embeddings' in self.analysis_results['model_metrics']:
            embeddings = self.analysis_results['model_metrics']['embeddings']
            context_lines.append(f"\nEmbeddings generated: {len(embeddings)} with dimension {len(embeddings[0])}")

        # Add knowledge graph information
        if 'knowledge_graph' in self.analysis_results['model_metrics']:
            graph = nx.node_link_graph(self.analysis_results['model_metrics']['knowledge_graph'])
            context_lines.append(f"\nKnowledge Graph: {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

        # Combine all lines into a single string
        context = "\n".join(context_lines)
        return context

    def save_results(self, dataset_name):
        """Save the analysis results to a JSON file in the results directory."""
        result_file_path = os.path.join(self.results_dir, f"{dataset_name}_results.json")
        
        # Convert analysis results to a JSON-serializable format
        serializable_results = numpy_to_python(self.analysis_results)
        
        with open(result_file_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        return result_file_path

    def generate_insights(self, results):
        insights = []
        top_feature = results['feature_importance'][0][0]
        insights.append(f"The most important factor affecting sales is {top_feature}.")
        # Additional insights logic
        return insights
    
    def save_best_model(self, results, dataset, target_column):
        # Filter out non-model metrics (like embeddings, knowledge_graph)
        model_metrics = {k: v for k, v in results.items() if isinstance(v, dict) and 'r2' in v}

        if not model_metrics:
            raise ValueError("No valid model metrics found in results.")

        # Find the best model based on R2 score
        best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['r2'])
        best_model = self.models[best_model_name]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset}_{target_column}_{best_model_name}_{timestamp}.joblib"
        filepath = os.path.join(self.models_dir, filename)
        
        model_data = {
            'model': best_model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'model_type': best_model_name,
            'dataset': dataset,
            'target_column': target_column
        }
        
        joblib.dump(model_data, filepath)
        
        return filename

    def load_model(self, filename):
        filepath = os.path.join(self.models_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filename} not found")
        
        return joblib.load(filepath)

    def perform_inference(self, model_filename, input_data):
        model_data = self.load_model(model_filename)
        
        # Preprocess input data
        input_df = pd.DataFrame([input_data])
        preprocessed_data = model_data['preprocessor'].transform(input_df)
        
        # Perform inference
        prediction = model_data['model'].predict(preprocessed_data)
        
        # Create context embedding for this inference
        feature_importances = self.get_feature_importances(model_data['model'], model_data['model_type'])
        embedding = self.create_context_embedding(prediction, None, feature_importances)
        
        return {
            'prediction': prediction.tolist(),
            'embedding': embedding.tolist()
        }

    def get_feature_importances(self, model, model_type):
        if model_type == 'xgboost':
            return dict(zip(self.feature_names, model.feature_importances_))
        elif model_type == 'neural_network':
            # Placeholder for neural network feature importance
            return {feature: 1.0 for feature in self.feature_names}
        else:  # ensemble
            xgb_importances = dict(zip(self.feature_names, self.models['xgboost'].feature_importances_))
            nn_importances = {feature: 1.0 for feature in self.feature_names}
            return {feature: (xgb_importances[feature] + nn_importances[feature]) / 2 
                    for feature in self.feature_names}

    def create_context_embedding(self, predictions, actual_values=None, feature_importances=None):
        # Summary statistics of predictions
        pred_stats = {
            'mean': np.mean(predictions),
            'median': np.median(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions)
        }

        # Model performance metrics (if actual values are provided)
        if actual_values is not None:
            mse = mean_squared_error(actual_values, predictions)
            mae = mean_absolute_error(actual_values, predictions)
            r2 = r2_score(actual_values, predictions)
        else:
            mse = mae = r2 = 0  # Placeholder values for inference

        # Top 5 feature importances (if provided)
        if feature_importances:
            top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_features = [(f, 0) for f in range(5)]  # Placeholder values

        # Combine all information into a single vector
        embedding = [
            pred_stats['mean'], pred_stats['median'], pred_stats['std'], pred_stats['min'], pred_stats['max'],
            mse, mae, r2
        ]
        embedding.extend([imp for _, imp in top_features])

        # Normalize the embedding
        scaler = MinMaxScaler()
        normalized_embedding = scaler.fit_transform(np.array(embedding).reshape(1, -1))[0]

        return normalized_embedding

    def get_saved_models(self):
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
        models = []
        for filename in model_files:
            model_data = self.load_model(filename)
            models.append({
                'filename': filename,
                'dataset': model_data['dataset'],
                'target_column': model_data['target_column'],
                'model_type': model_data['model_type']
            })
        return models
    
    def generate_tabular_embeddings(self, X):
        embeddings = []
        for _, row in X.iterrows():
            text_representation = " ".join(f"{col}: {val}" for col, val in row.items())
            inputs = self.bert_tokenizer(text_representation, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(normalize(embedding.reshape(1, -1))[0])
        return np.array(embeddings)

    def preprocess_for_knowledge_graph(self, df, target_column):
        def safe_float_convert(x):
            try:
                return float(str(x).replace('$', '').replace(',', '').strip())
            except ValueError:
                return np.nan

        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Handle numeric columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_features:
            X[col] = X[col].apply(safe_float_convert)

        # Handle categorical columns
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))

        # Handle NaN values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        return X, y

    def get_feature_importances_for_graph(self, X, y):
        importances = {}
        for feature in X.columns:
            correlation, _ = spearmanr(X[feature], y)
            importances[feature] = abs(correlation) if not np.isnan(correlation) else 0
        return importances

    def build_knowledge_graph(self, X, y, feature_importances):
        self.knowledge_graph = nx.Graph()

        # Add nodes for features
        for feature, importance in feature_importances.items():
            self.knowledge_graph.add_node(feature, type='feature', importance=importance)

        # Add node for target variable
        self.knowledge_graph.add_node(y.name, type='target')

        # Add edges between features and target
        for feature in X.columns:
            self.knowledge_graph.add_edge(feature, y.name, weight=feature_importances.get(feature, 0))

        # Add edges between features
        for i, feature1 in enumerate(X.columns):
            for feature2 in X.columns[i+1:]:
                correlation, _ = spearmanr(X[feature1], X[feature2])
                if not np.isnan(correlation) and abs(correlation) > 0.5:
                    self.knowledge_graph.add_edge(feature1, feature2, weight=abs(correlation))

    def build_and_save_knowledge_graph(self, dataset, target_column):
        try:
            graph_file = os.path.join(self.results_dir, f"{dataset}_{target_column}_knowledge_graph.json")
            
            # Try to load existing graph, if it exists
            if os.path.exists(graph_file):
                with open(graph_file, 'r') as f:
                    return json.load(f)
            
            # If the file doesn't exist, create the graph
            df = pd.read_csv(os.path.join(DATASET_FOLDER, dataset))
            X, y = self.preprocess_for_knowledge_graph(df, target_column)
            
            feature_importances = self.get_feature_importances_for_graph(X, y)
            self.build_knowledge_graph(X, y, feature_importances)
            
            # Convert the graph data to a JSON-serializable format
            graph_data = nx.node_link_data(self.knowledge_graph)
            
            # Convert all numpy types to Python types
            def convert_numpy(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {convert_numpy(key): convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj

            graph_data = convert_numpy(graph_data)
            
            # Save the knowledge graph to disk
            with open(graph_file, 'w') as f:
                json.dump(graph_data, f)
            
            return graph_data
        except Exception as e:
            print(f"Error building knowledge graph: {str(e)}")
            return None

    def get_knowledge_graph_data(self, dataset, target_column):
        try:
            return self.build_and_save_knowledge_graph(dataset, target_column)
        except Exception as e:
            print(f"Error getting knowledge graph data: {str(e)}")
            return None

    def summarize_dataframe(self, df):
        """
        Generates a summary of the dataframe.
        
        Args:
        df (pandas.DataFrame): The dataframe to summarize.
        
        Returns:
        str: A string containing a summary of the dataframe.
        """
        summary = []
        
        # Basic dataframe info
        summary.append(f"Number of rows: {df.shape[0]}")
        summary.append(f"Number of columns: {df.shape[1]}")
        
        # Column types
        type_counts = df.dtypes.value_counts()
        summary.append("Column types:")
        for dtype, count in type_counts.items():
            summary.append(f"  {dtype}: {count}")
        
        # Numeric columns summary
        numeric_columns = df.select_dtypes(include=['int64', 'float64'])
        if not numeric_columns.empty:
            summary.append("\nNumeric Columns Summary:")
            for column in numeric_columns.columns:
                summary.append(f"  {column}:")
                summary.append(f"    Mean: {df[column].mean():.2f}")
                summary.append(f"    Median: {df[column].median():.2f}")
                summary.append(f"    Std Dev: {df[column].std():.2f}")
                summary.append(f"    Min: {df[column].min():.2f}")
                summary.append(f"    Max: {df[column].max():.2f}")
        
        # Categorical columns summary
        categorical_columns = df.select_dtypes(include=['object', 'category'])
        if not categorical_columns.empty:
            summary.append("\nCategorical Columns Summary:")
            for column in categorical_columns.columns:
                summary.append(f"  {column}:")
                summary.append(f"    Unique values: {df[column].nunique()}")
                summary.append(f"    Top 3 values: {', '.join(df[column].value_counts().nlargest(3).index.astype(str))}")
        
        # Missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            summary.append("\nMissing Values:")
            for column, count in missing_data[missing_data > 0].items():
                summary.append(f"  {column}: {count} ({count/len(df):.2%})")
        
        return "\n".join(summary)

tabular_ml_processor = TabularMLProcessor()