# AI-SOLUTIONS-FOR-FARMER
#app.py
# app.py
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from crop_price_prediction import model_pipeline as price_model
from crop_yield_prediction import model as yield_model
from weather_prediction import WeatherPredictor
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize weather predictor
weather_predictor = WeatherPredictor('seattle-weather.csv')
weather_predictor.build_model()
weather_predictor.train_with_cross_validation()

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    name = request.form['name']
    mobile = request.form['mobile']
    if len(mobile) == 10 and mobile.isdigit():
        session['user'] = name
        return redirect(url_for('dashboard'))
    return render_template('login.html', error="Please enter a valid 10-digit mobile number")

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/predict/<model_type>', methods=['GET', 'POST'])
def predict(model_type):
    if 'user' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            if model_type == 'price':
                # Existing price prediction code
                input_data = pd.DataFrame({
                    'State': [request.form['state']],
                    'Crop': [request.form['crop']],
                    'CostCultivation': [float(request.form['cost_cultivation'])],
                    'CostCultivation2': [float(request.form['cost_cultivation2'])],
                    'Production': [float(request.form['production'])],
                    'Yield': [float(request.form['yield_val'])],
                    'Temperature': [float(request.form['temperature'])],
                    'RainFall Annual': [float(request.form['rainfall'])]
                })
                prediction = price_model.predict(input_data)[0]
                return render_template('result.html', result=f"Predicted Price: ₹{prediction:.2f}")

            elif model_type == 'yield':
                # Existing yield prediction code
                input_data = pd.DataFrame({
                    'Crop': [request.form['crop']],
                    'Nitrogen': [float(request.form['nitrogen'])],
                    'Phosphorus': [float(request.form['phosphorus'])],
                    'Potassium': [float(request.form['potassium'])],
                    'Temperature': [float(request.form['temperature'])],
                    'Humidity': [float(request.form['humidity'])],
                    'pH_Value': [float(request.form['ph'])],
                    'Rainfall': [float(request.form['rainfall'])]
                })
                prediction = yield_model.predict(input_data)[0]
                return render_template('result.html', result=f"Predicted Yield: {prediction:.2f} KG/hectare")

            elif model_type == 'weather':
                # Updated weather prediction code
                input_data = []
                for i in range(7):
                    # Get basic features from form
                    data_point = {
                        'precipitation': float(request.form[f'precip_{i}']),
                        'temp_max': float(request.form[f'temp_max_{i}']),
                        'temp_min': float(request.form[f'temp_min_{i}']),
                        'wind': float(request.form[f'wind_{i}']),
                        'month': int(request.form[f'month_{i}']),
                        'day_of_week': int(request.form[f'day_{i}'])
                    }
                    
                    # Calculate derived features
                    data_point['temp_range'] = data_point['temp_max'] - data_point['temp_min']
                    data_point['temp_avg'] = (data_point['temp_max'] + data_point['temp_min']) / 2
                    
                    input_data.append(data_point)
                
                # Convert to DataFrame for easier calculations
                df = pd.DataFrame(input_data)
                
                # Calculate rolling features
                df['rolling_temp_avg'] = df['temp_avg'].rolling(window=3, min_periods=1).mean()
                df['rolling_precip'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
                df['rolling_wind'] = df['wind'].rolling(window=3, min_periods=1).mean()
                
                # Fill NaN values from rolling calculations
                df.fillna(method='bfill', inplace=True)
                df.fillna(method='ffill', inplace=True)
                
                # Ensure columns are in the correct order
                required_features = [
                    'precipitation', 'temp_max', 'temp_min', 'wind', 
                    'month', 'day_of_week', 'temp_range', 'temp_avg',
                    'rolling_temp_avg', 'rolling_precip', 'rolling_wind'
                ]
                
                input_df = df[required_features]
                
                # Get prediction
                result = weather_predictor.predict_weather(input_df)
                
                # Display results
                return render_template('result.html', 
                                    result=f"Weather Prediction: {result['primary_prediction']}",
                                    confidence=f"Confidence: {result['confidence']:.2f}")

        except Exception as e:
            return render_template('result.html', error=f"Error: {str(e)}")

    return render_template(f'{model_type}_form.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
    #weather.py
    import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class WeatherPredictor:
    def __init__(self, data_path, window_size=7):
        self.window_size = window_size
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.le = LabelEncoder()
        self.history = None
        self.load_and_process_data(data_path)

    def load_and_process_data(self, data_path):
        """Load and preprocess the weather data"""
        self.data = pd.read_csv(data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Feature engineering
        self.data['month'] = self.data['date'].dt.month
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['temp_range'] = self.data['temp_max'] - self.data['temp_min']
        self.data['temp_avg'] = (self.data['temp_max'] + self.data['temp_min']) / 2
        
        # Create rolling features
        self.data['rolling_temp_avg'] = self.data['temp_avg'].rolling(window=3).mean()
        self.data['rolling_precip'] = self.data['precipitation'].rolling(window=3).sum()
        self.data['rolling_wind'] = self.data['wind'].rolling(window=3).mean()
        
        # Handle missing values from rolling calculations
        self.data.fillna(method='bfill', inplace=True)
        
        # Encode weather labels
        self.data['weather_encoded'] = self.le.fit_transform(self.data['weather'])
        self.weather_onehot = to_categorical(self.data['weather_encoded'])
        
        # Scale features
        self.features = ['precipitation', 'temp_max', 'temp_min', 'wind', 
                        'month', 'day_of_week', 'temp_range', 'temp_avg',
                        'rolling_temp_avg', 'rolling_precip', 'rolling_wind']
        self.scaled_features = self.scaler.fit_transform(self.data[self.features])

    def create_sequences(self):
        """Create sequences for LSTM with enhanced features"""
        X, y = [], []
        for i in range(self.window_size, len(self.scaled_features)):
            X.append(self.scaled_features[i-self.window_size:i])
            y.append(self.weather_onehot[i])
        return np.array(X), np.array(y)

    def build_model(self):
        """Build enhanced LSTM model"""
        self.model = Sequential([
            Bidirectional(LSTM(256, return_sequences=True), 
                         input_shape=(self.window_size, len(self.features))),
            BatchNormalization(),
            Dropout(0.4),
            
            Bidirectional(LSTM(128, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.4),
            
            Bidirectional(LSTM(64)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            
            Dense(len(self.le.classes_), activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer,
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    def train_with_cross_validation(self, n_splits=5):
        """Train model with time series cross-validation"""
        X, y = self.create_sequences()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        histories = []
        fold_accuracies = []
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nTraining Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            histories.append(history.history)
            fold_accuracies.append(max(history.history['val_accuracy']))
        
        self.history = histories
        return np.mean(fold_accuracies)

    def predict_weather(self, input_data):
        """Make weather prediction with confidence scores"""
        if len(input_data) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} days of data")
            
        # Scale input data
        scaled_input = self.scaler.transform(input_data)
        sequence = scaled_input[-self.window_size:].reshape(1, self.window_size, len(self.features))
        
        # Get predictions and probabilities
        pred_probs = self.model.predict(sequence, verbose=0)
        pred_class = np.argmax(pred_probs, axis=1)
        confidence = pred_probs[0][pred_class[0]]
        
        # Get top 3 predictions with probabilities
        top_3_indices = np.argsort(pred_probs[0])[-3:][::-1]
        top_3_predictions = []
        for idx in top_3_indices:
            weather_type = self.le.inverse_transform([idx])[0]
            probability = pred_probs[0][idx]
            top_3_predictions.append((weather_type, probability))
        
        return {
            'primary_prediction': self.le.inverse_transform(pred_class)[0],
            'confidence': confidence,
            'top_3_predictions': top_3_predictions
        }

    def plot_training_history(self):
        """Plot training history with cross-validation results"""
        fig = go.Figure()
        
        # Plot accuracy for each fold
        for fold, history in enumerate(self.history):
            fig.add_trace(go.Scatter(
                y=history['accuracy'],
                name=f'Training Accuracy Fold {fold+1}',
                line=dict(width=1, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                y=history['val_accuracy'],
                name=f'Validation Accuracy Fold {fold+1}',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Model Training History Across All Folds',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            template='plotly_white'
        )
        return fig

    def plot_feature_importance(self):
        """Plot feature importance using permutation importance"""
        X, y = self.create_sequences()
        base_score = self.model.evaluate(X, y, verbose=0)[1]
        importance_scores = []
        
        for i in range(len(self.features)):
            X_permuted = X.copy()
            X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i])
            permuted_score = self.model.evaluate(X_permuted, y, verbose=0)[1]
            importance = base_score - permuted_score
            importance_scores.append(importance)
        
        fig = px.bar(
            x=self.features,
            y=importance_scores,
            title='Feature Importance',
            labels={'x': 'Features', 'y': 'Importance Score'}
        )
        return fig

def get_user_input(features):
    """Get weather data input from user"""
    print("\nPlease enter weather data for the last 7 days (most recent first):")
    input_data = []
    
    for day in range(7):
        print(f"\nDay {7-day}:")
        try:
            data_point = {
                'precipitation': float(input("Precipitation (mm): ")),
                'temp_max': float(input("Maximum Temperature (°C): ")),
                'temp_min': float(input("Minimum Temperature (°C): ")),
                'wind': float(input("Wind Speed (m/s): ")),
                'month': int(input("Month (1-12): ")),
                'day_of_week': int(input("Day of Week (0-6, 0=Monday): "))
            }
            
            # Calculate derived features
            data_point['temp_range'] = data_point['temp_max'] - data_point['temp_min']
            data_point['temp_avg'] = (data_point['temp_max'] + data_point['temp_min']) / 2
            
            if day < 5:  # Can only calculate rolling averages after first two days
                data_point['rolling_temp_avg'] = data_point['temp_avg']
                data_point['rolling_precip'] = data_point['precipitation']
                data_point['rolling_wind'] = data_point['wind']
            else:
                prev_points = input_data[-2:]  # Get last two points for rolling calcs
                data_point['rolling_temp_avg'] = np.mean([p['temp_avg'] for p in prev_points + [data_point]])
                data_point['rolling_precip'] = np.sum([p['precipitation'] for p in prev_points + [data_point]])
                data_point['rolling_wind'] = np.mean([p['wind'] for p in prev_points + [data_point]])
            
            input_data.append(data_point)
        except ValueError:
            print("Invalid input! Please enter numerical values.")
            return None
    
    # Convert to DataFrame in correct feature order
    df = pd.DataFrame(input_data)[features]
    return df.values

def main():
    # Initialize predictor
    predictor = WeatherPredictor('seattle-weather.csv')
    predictor.build_model()
    
    # Train model with cross-validation
    mean_accuracy = predictor.train_with_cross_validation()
    print(f"\nModel Accuracy: {mean_accuracy:.4f}")
    
    # Plot training history and feature importance
    history_fig = predictor.plot_training_history()
    importance_fig = predictor.plot_feature_importance()
    history_fig.show()
    importance_fig.show()
    
    while True:
        print("\n=== Enhanced Weather Prediction System ===")
        print("1. Enter new weather data for prediction")
        print("2. View model accuracy and visualizations")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            input_data = get_user_input(predictor.features)
            if input_data is not None:
                result = predictor.predict_weather(input_data)
                print(f"\nPrimary Prediction: {result['primary_prediction']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print("\nTop 3 Predictions:")
                for weather, prob in result['top_3_predictions']:
                    print(f"{weather}: {prob:.2f}")
        
        elif choice == '2':
            print(f"\nModel Accuracy: {mean_accuracy:.4f}")
            print("Training history and feature importance plots have been displayed")
            
        elif choice == '3':
            print("\nThank you for using the Enhanced Weather Prediction System!")
            break
        
        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()
#crop price.py
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd

# Load dataset
file_path  = "dataset.csv"
  # Update with your file path
dataset = pd.read_csv(file_path)

# Prepare features and target
X = dataset.drop(columns=['Price'])
y = dataset['Price']

# Preprocess categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['State', 'Crop'])
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessor and linear regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model_pipeline.fit(X_train, y_train)

# Define the prediction loop function
def PricePredictor(model):
    while True:
        try:
            user_choice = int(input("Would you like to make a prediction? Type '1' for prediction or '2' to exit: ").strip())
        except ValueError:
            print("Invalid choice! Please type '1' to continue or '2' to quit.")
            continue

        if user_choice == 2:
            print("Exiting the prediction loop.")
            break
        elif user_choice == 1:
            try:
                # Collect inputs for prediction
                state = input("Enter the state: ").strip().capitalize()
                crop = input("Enter the crop: ").strip().capitalize()
                cost_cultivation = float(input("Enter the Cost of Cultivation: "))
                cost_cultivation2 = float(input("Enter the Secondary Cost of Cultivation: "))
                production = float(input("Enter the Production value: "))
                yield_val = float(input("Enter the Yield value: "))
                temperature = float(input("Enter the Temperature: "))
                rainfall_annual = float(input("Enter the Annual Rainfall: "))

                # Organize input into a DataFrame to match training format
                input_data = pd.DataFrame({
                    'State': [state],
                    'Crop': [crop],
                    'CostCultivation': [cost_cultivation],
                    'CostCultivation2': [cost_cultivation2],
                    'Production': [production],
                    'Yield': [yield_val],
                    'Temperature': [temperature],
                    'RainFall Annual': [rainfall_annual]
                })

                # Predict price
                predicted_price = model.predict(input_data)
                print(f"Predicted Price for the given inputs: {predicted_price[0]:.2f}")
            
            except ValueError:
                print("Invalid input! Please enter numeric values where applicable.")
        else:
            print("Invalid choice! Please type '1' to continue or '2' to quit.")

# Run the prediction loop
PricePredictor(model_pipeline)
#crop yield prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
data = pd.read_csv('Crop_Yield_Prediction.csv')

# Specify features and target variable
X = data.drop('Yield', axis=1)
y = data['Yield']

# Identify categorical and numerical features
categorical_features = ['Crop']
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Full model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Get the unique list of crops for validation
valid_crops = X['Crop'].str.title().unique()

# Function to get user input and predict yield
def YieldPredictor():
    print("Please enter the following details to predict crop yield:")

    # Validate crop input
    crop = input("Crop (e.g., Rice, Wheat): ").strip().title()
    if crop not in valid_crops:
        print("Invalid crop name. Please enter a valid crop from the dataset.")
        return

    try:
        # Collect other inputs with error handling for invalid values
        nitrogen = float(input("Nitrogen (N): "))
        phosphorus = float(input("Phosphorus (P): "))
        potassium = float(input("Potassium (K): "))
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        pH_value = float(input("pH Value: "))
        rainfall = float(input("Rainfall (mm): "))

        # Create a DataFrame for the new input
        new_input = pd.DataFrame({
            'Crop': [crop],
            'Nitrogen': [nitrogen],
            'Phosphorus': [phosphorus],
            'Potassium': [potassium],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'pH_Value': [pH_value],
            'Rainfall': [rainfall]
        })

        # Predict yield
        predicted_yield = model.predict(new_input)
        print(f'Predicted Yield for {crop}: {predicted_yield[0]}')

    except ValueError:
        print("Invalid input. Please enter numeric values for Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH Value, and Rainfall.")

# Main loop
while True:
    print("\nWould you like to:")
    print("1. Predict crop yield")
    print("2. Exit")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        YieldPredictor()
    elif choice == '2':
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")
