import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


#yfinance 라이브러리(증권 데이터 수집)로 삼전의 데이터를 불러옴
ijaeyong = yf.download('005930.KS', start='2010-10-16', end='2022-01-01')
ijaeyong_END = yf.download('005930.KS', start='2010-10-16', end='2023-01-01')

#데이터 정규화
ijaeyong = ijaeyong['Close']
ijaeyong = ijaeyong.fillna(method='bfill') 
scaler = MinMaxScaler()
ijaeyong = scaler.fit_transform(np.array(ijaeyong).reshape(-1, 1))

ijaeyong_END = ijaeyong_END['Close']
ijaeyong_END = ijaeyong_END.fillna(method='bfill') 
scaler = MinMaxScaler()
ijaeyong_END = scaler.fit_transform(np.array(ijaeyong_END).reshape(-1, 1))

#입력 / 출력 데이터 생성
sequence_length = 30
X, y = [], []
for i in range(len(ijaeyong) - sequence_length):
    X.append(ijaeyong[i:i+sequence_length])
    y.append(ijaeyong[i+sequence_length])
X = np.array(X)
y = np.array(y)

#tensorflow LSTM 모델 정의
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#뺑뺑이 횟수 정하기
model.fit(X, y, epochs=10, batch_size=10000)
#X= 입력 데이터
#Y= 출력 데이터

#뻉뺑이의 결과값(미래의 값)을 배출하는데, 배출할 일자 수(30이면 30일 뒤까지)를 정함
future_days = 365  
last_sequence = X[-1]  
future_predictions = []

#학습 데이터를 기반으로 미래 주가를 뽑아냄
for _ in range(future_days):
    #아까 예측한 값을 저장함
    prediction = model.predict(last_sequence.reshape(1, sequence_length, 1))
    #예측된 값 저장
    future_predictions.append(prediction[0, 0])
    #위에 리스트에서 가장 오래된 값을 버림
    last_sequence = np.roll(last_sequence, shift=-1)
    #새로 예측된 값을 리스트에 넣음
    last_sequence[-1] = prediction[0, 0]

#최종적으로 원래 주가와 예측 주가를 합쳐서 그래프에 나타냄.
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(ijaeyong_END)), ijaeyong_END, label='Actual Prices')
plt.plot(np.arange(len(ijaeyong), len(ijaeyong) + len(future_predictions)), future_predictions, label='Future Predictions', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Scaled Price')
plt.legend()
plt.show()


#!! 해당 프로그램은 단순히 "이전 주가를 기준으로 반복 학습하여" 미래를 예측하는 것으로, 정확하지 않음.