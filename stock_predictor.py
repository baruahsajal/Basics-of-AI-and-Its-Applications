import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class StockPricePredictor:
    def __init__(self, ticker='AAPL', period='2y'):
        self.ticker = ticker
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        
    def fetch_data(self):
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, period=self.period)
        return self.data
    
    def prepare_data(self, lookback=60):
        dataset = self.data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(dataset)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        train_size = int(len(X) * 0.8)
        return X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    
    def build_model(self, lookback=60):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
    
    def predict_and_plot(self, X_test, y_test):
        predictions = self.scaler.inverse_transform(self.model.predict(X_test))
        actuals = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        plt.figure(figsize=(10, 5))
        plt.plot(actuals, color='cyan', label='Actual Price')
        plt.plot(predictions, color='magenta', label='Predicted Price')
        plt.title(f'{self.ticker} Stock Prediction using LSTM')
        plt.legend()
        plt.savefig('lstm_predictions.png', transparent=True)
        plt.show()

if __name__ == '__main__':
    predictor = StockPricePredictor(ticker='GOOGL')
    predictor.fetch_data()
    X_train, X_test, y_train, y_test = predictor.prepare_data()
    predictor.build_model()
    predictor.train(X_train, y_train)
    predictor.predict_and_plot(X_test, y_test)
