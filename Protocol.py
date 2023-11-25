# 필요한 라이브러리 임포트
import yfinance as yf  # 주식 데이터 다운로드에 사용
import datetime  # 날짜 및 시간 관련 작업에 사용
import matplotlib.pyplot as plt  # 데이터 시각화에 사용
import pmdarima as pm  # ARIMA 모델 자동 선택을 위한 라이브러리
import numpy as np  # 수치 계산에 사용
import pandas as pd  # 데이터 조작 및 분석에 사용
import math  # 수학 함수에 사용
import exchange_calendars as ecals  # 거래소 달력 정보를 얻기 위한 라이브러리
from pmdarima.arima.utils import ndiffs  # ARIMA 차분 계산을 위한 함수
from statsmodels.tsa.arima_model import ARIMA  # ARIMA 모델 구현에 사용
from statsmodels.tsa.stattools import adfuller  # ADF 테스트를 위한 함수
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 성능 평가에 사용

# 주식 데이터 다운로드 함수
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# 거래량이 0인 날짜의 Adjusted Close 값 제거 함수
def remove_zero_volume(data):
    return data['Adj Close'][data['Volume'] != 0]

# ADF(Augmented Dickey-Fuller) 테스트 함수
def adf_test(data):
    # ADF 테스트를 적용하고 결과를 저장
    result = adfuller(data.values)

    # ADF 테스트 결과 출력
    # ADF 통계량 = 단위근을 검정하는 통계치
    print('ADF Statistics: %f' % result[0])
    # 귀무가설(어떤 효과나 차이가 없다고 가정하는 가설)이 참일 때, 현재의 데이터가 더 극단적인 데이터를 얻을 확률
    print('p-value: %f' % result[1])
    # 시계열 데이터의 변화를 확인하기 위해 이전 값들과의 차이를 확인, 몇 번째 이전 값까지 확인했는지 나타냄
    print('num of lags: %f' % result[2])
    # 분석에 사용된 데이터 수
    print('num of observations: %f' % result[3])
    # 귀무가설을 평가하는데 사용되는 임계치
    print('Critical values:')
    # ADF 테스트에 사용되는 임계치를 담은 딕셔너리. 각 항목은 검정 통계량에 대한 임계치를 나타냄
    for k, v in result[4].items():
        print('\t%s: %.3f' % (k, v))

# 이동평균 및 표준 편차를 시각화하는 함수
def plot_rolling(data, interval):
    rolmean = data.rolling(interval).mean()
    rolstd = data.rolling(interval).std()
    plt.figure(figsize=(10, 6))
    plt.xlabel('Date')
    orig = plt.plot(data, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label=f'Rolling Mean {interval}')
    std = plt.plot(rolstd, color='black', label=f'Rolling Std {interval}')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

# ADF 테스트 및 차분 횟수 결정 함수
def adf_and_diff(data):
    adf_result = adfuller(data.values)
    print('ADF Statistics:', adf_result[0])
    print('p-value:', adf_result[1])
    print('num of lags:', adf_result[2])
    print('num of observations:', adf_result[3])
    print('Critical values:')
    for k, v in adf_result[4].items():
        print('\t%s: %.3f' % (k, v))

    n_diffs = ndiffs(data, alpha=0.05, test='adf', max_d=6)
    print(f"Estimated differencing order (d): {n_diffs}")
    return n_diffs

# 자동 ARIMA 모델 훈련 함수
def auto_arima_model(train_data, n_diffs):
    model_fit = pm.auto_arima(
        y=train_data,
        d=n_diffs,
        start_p=0, max_p=2,
        start_q=0, max_q=2,
        m=1, seasonal=False,
        stepwise=True,
        trace=True
    )
    print(model_fit.summary())
    return model_fit

# n 기간 동안의 예측과 신뢰 구간 계산 함수
def forecast_n_step(model, n=1):
    fc, conf_int = model.predict(n_periods=n, return_conf_int=True)
    return fc.tolist()[0:n], np.asarray(conf_int).tolist()[0:n]

# len 기간 동안의 전체 예측과 신뢰 구간 계산 함수
def forecast(len, model, index, data=None):
    y_pred = []
    pred_upper = []
    pred_lower = []

    if data is not None:
        for new_ob in data:
            fc, conf = forecast_n_step(model)
            y_pred.append(fc[0])
            pred_upper.append(conf[0][1])
            pred_lower.append(conf[0][0])
            model.update(new_ob)
    else:
        for i in range(len):
            fc, conf = forecast_n_step(model)
            y_pred.append(fc[0])
            pred_upper.append(conf[0][1])
            pred_lower.append(conf[0][0])
            model.update(fc[0])
    return pd.Series(y_pred, index=index), pred_upper, pred_lower

# 주식 거래일을 얻는 함수
def get_open_dates(start, end):
    k = ecals.get_calendar("XKRX")
    df = pd.DataFrame(k.schedule.loc[start:end])
    date_list = [i.strftime("%Y-%m-%d") for i in df['open']]
    date_index = pd.DatetimeIndex(date_list)
    return date_index

# 메인 함수
def main():
    # 시작 및 종료 날짜 설정
    start_date = datetime.datetime(2012, 6, 1)
    end_date = datetime.datetime(2022, 6, 1)

    # 주식 데이터 다운로드
    ticker = '005930.KS'
    samsung = download_stock_data(ticker, start_date, end_date)

    # 다운로드한 주식 데이터 출력
    print("주가 데이터>\n")
    print(samsung)
    print('\n')
    print(samsung.info())

    # 거래량이 0인 날짜의 Adjusted Close 값 제거
    data = remove_zero_volume(samsung)

    # ADF 테스트 및 결과 출력
    print('테스트 결과')
    adf_test(data)
    
    # ADF 테스트를 통한 차분 횟수 결정
    n_diffs = adf_and_diff(data)
    
    # 훈련 및 테스트 데이터 분리
    train_data, test_data = data[:int(len(data) * 0.9)], data[int(len(data) * 0.9):]
    
    # 이동평균 시각화
    plot_rolling(data, 20)
    
    # 자동 ARIMA 모델 훈련
    model_fit = auto_arima_model(train_data, n_diffs)
    
    # 테스트 데이터에 대한 예측 및 시각화
    fc, upper, lower = forecast(len(test_data), model_fit, test_data.index, data=test_data)
    lower_series = pd.Series(lower, index=test_data.index)
    upper_series = pd.Series(upper, index=test_data.index)
    plt.figure(figsize=(20, 6))
    plt.plot(train_data, label='training data')
    plt.plot(test_data, c='b', label='Test data (real price)')
    plt.plot(fc, c='r', label='predicted price')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
    plt.legend(loc='upper left')
    plt.show()

    # 1년 후 주식 가격 예측
    date_index = get_open_dates("2022-06-02", "2023-06-01")
    fc2, upper2, lower2 = forecast(len(date_index), model_fit, date_index)
    print('1년 후 주가')
    print(fc2.tail())
    lower_series2 = pd.Series(lower2, index=date_index)
    upper_series2 = pd.Series(upper2, index=date_index)

    # 1년 후 예측 결과 시각화
    plt.figure(figsize=(20, 6))
    plt.plot(train_data, label='source data')
    plt.plot(test_data, c='b', label='actual price')
    plt.plot(fc, c='r', label='predicted price')
    plt.plot(fc2, c='g', label='predicted price after a year')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
    plt.fill_between(lower_series2.index, lower_series2, upper_series2, color='k', alpha=.10)
    plt.title('year from now')
    plt.legend(loc='upper left')
    plt.show()

# 스크립트가 직접 실행되면 main 함수 호출
if __name__ == "__main__":
    main()
