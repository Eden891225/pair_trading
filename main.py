import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arch.unitroot import ADF
import statsmodels.api as sm

#建構配對交易類別
class pair_trading:
  def adf(price_x, price_y, start, end): #ADF檢定函數
    if price_x is None or price_y is None:
      print('缺少價格序列')
    price_x = price_x[start:end] #樣本內價格
    price_y = price_y[start:end]
    price_x = np.log(price_x).diff()[1:] #對數價格
    price_y = np.log(price_y).diff()[1:]
    adf_x = ADF(price_x) #執行ADF檢定
    adf_y = ADF(price_y)
    print(adf_x.summary) #顯示檢定結果
    print(adf_y.summary)
  def cointeration(price_x, price_y, start, end): #共整合檢定函數
    if price_x is None or price_y is None:
      print('缺少價格序列')
    price_x = price_x[start:end] #樣本內價格
    price_y = price_y[start:end]
    price_x = np.log(price_x) #對數價格
    price_y = np.log(price_y)
    result = sm.OLS(price_y, sm.add_constant(price_x)).fit() #將對數價格跑回歸
    resid = result.resid #取出模型殘差
    resid.plot() #畫出殘插圖
    adf = ADF(resid) #將殘差做ADF檢定
    if adf.pvalue >= 0.05: #p-value大於5%為顯著，無法拒絕虛無假設
      print('交易不具共整合關係')
      print(f'P-value: {adf.pvalue}') #印出p-value
      print(f'Alpha: {result.params[0]}') #印出Alpha
      print(f'Beta: {result.params[1]}') #印出Beta
      return None, None #不回傳
    else: #拒絕虛無假設
      print('交易具共整合關係')
      print(f'P-value: {adf.pvalue}')
      print(f'Alpha: {result.params[0]}')
      print(f'Beta: {result.params[1]}')
      return result.params[0], result.params[1] #回傳Alpha和Beta
  def trade_signal(price_x, price_y, alpha, beta, start, end, test_start, test_end, multiplier, stop):
    #交易訊號函數
    if price_x is None or price_y is None:
      print('缺少價格序列')
    price_x = price_x[start:test_end] #全樣本價格
    price_y = price_y[start:test_end]
    price_x = np.log(price_x) #對數價格
    price_y = np.log(price_y)
    spread = price_y - alpha - beta * price_x #算出殘差(價差)
    std = spread[start:end].std() #算出殘差標準差
    level = (float('-inf'), -stop*std, -multiplier*std, 0, multiplier*std, stop*std, float('inf')) #將價差分為6個區間
    signal = pd.cut(spread[start:test_end], level, labels=False) #回傳各區間代表的值
    print(level) #印出區間界線值
    return signal, spread #回傳交易訊號和殘差
  def trade_account(price_x, price_y, signal, beta, spread, start, end, test_start, test_end, fund=1000000):
    #交易帳戶函數
    position, number_x, number_y, interest_income = 0, 0, 0, 0 #初始化部位、持有X、Y股票數量和融券利息收入為0
    equity = fund #初始化權益數等於起始資金
    equitys, cum_return, buy, sell, offset = pd.Series(0.0, index=signal.index), pd.Series(0.0, index=signal.index), pd.Series(0.0, index=signal.index), pd.Series(0.0, index=signal.index), pd.Series(0.0, index=signal.index)
    #初始化權益數序列、累積報酬率序列、買入點(買Y賣X)序列、賣出點(賣Y買X)序列和平倉序列為0
    equitys[signal.index[0]] = equity #初始化第1天的權益數
    cum_return[signal.index[0]] = 1 #初始化第1天的累積報酬率為1
    buy[signal.index[0]], sell[signal.index[0]], offset[signal.index[0]] = None, None, None #初始化第1天沒有買賣和平倉
    tax = 0.003 #證交稅
    commision = 0.001425 #交易手續費
    short_cost = 0.0008 #融券手續費(融券賣出時收)
    short_interest = 0.002 #融券保證金利率
    slip = 0.005 #滑價
    for i in range(0, len(signal.index)): #開始回測迴圈
      today = signal.index[i].strftime('%Y-%m-%d') #今日
      if today == end: #若今日等於樣本內最後1天
        if position == 1: #若有X長部位，Y空部位，明日開盤平倉
          open_x = price_x['開盤價(元)'][test_start] #明日(樣本外起始日)日收開盤價
          open_y = price_y['開盤價(元)'][test_start]
          equity = fund + number_x * open_x * (1 - commision - tax - slip) - number_y * open_y * (1 + commision + slip) + interest_income #計算平倉權益
          position, number_x, number_y, interest_income = 0, 0, 0, 0 #初始化部位、持有X、Y股票數量和融券利息收入為0
          fund = equity #將權益數全部變為資金
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[test_start] = equity #將明日權益數放入權益數序列
          cum_return[test_start] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[test_start], buy[test_start], offset[test_start] = None, None, spread[next_date] #設定平倉點
        elif position == -1: #若有Y長部位，X空部位，明日開盤平倉
          open_x = price_x['開盤價(元)'][test_start] #明日(樣本外起始日)日收開盤價
          open_y = price_y['開盤價(元)'][test_start]
          equity = fund + number_y * open_y * (1 - commision - tax - slip) - number_x * open_x * (1 + commision + slip) + interest_income #計算平倉權益
          position, number_x, number_y, interest_income = 0, 0, 0, 0 #初始化部位、持有X、Y股票數量和融券利息收入為0
          fund = equity #將權益數全部變為資金
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[test_start] = equity #將明日權益數放入權益數序列
          cum_return[test_start] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[test_start], buy[test_start], offset[test_start] = None, None, spread[test_start] #設定平倉點
        else: #沒有部位，不執行動作
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[test_start] = equity #將明日權益數放入權益數序列
          cum_return[test_start] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[test_start], buy[test_start], offset[test_start] = None, None, None #不設定
      elif today == test_end: #若今日為樣本外最後一天，不執行動作，跳出迴圈
        break
      else:
        next_date = signal.index[i+1].strftime('%Y-%m-%d') #明日
        if position == 0 and signal[i] == 4: #若沒部位，在交易訊號上界以上，停損線以下
          #買X賣Y
          position = 1 #設定為X長部位，Y空部位
          open_x = price_x['開盤價(元)'][next_date] #隔日開盤價
          open_y = price_y['開盤價(元)'][next_date]
          close_x = price_x['收盤價(元)'][next_date] #隔日收盤價
          close_y = price_y['收盤價(元)'][next_date]
          number_x = (fund * beta / (1 + beta)) // (open_x * (1 + commision + slip)) #買入幾股X
          number_y = (fund / (1 + beta)) // (open_y * (1 + commision + tax + short_cost + slip)) // 1000 * 1000 #放空幾張Y
          fund = fund - number_x * open_x * (1 + commision + slip) + number_y * open_y * (1 - commision - tax - short_cost - slip) #明日剩餘資金
          interest_income = open_y * 0.9 * short_interest / 365 #融券保證金利息收入
          equity = fund + number_x * close_x * (1 - commision - tax - slip) - number_y * close_y * (1 + commision + slip) + interest_income #以明日收盤價計算明日權益
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[next_date] = equity #將明日權益數放入權益數序列
          cum_return[next_date] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[next_date], buy[next_date], offset[next_date] = spread[next_date], None, None #設定賣出點
        elif position == 1 and (signal[i] == 4 or signal[i] == 3): #若有X長部位，Y空部位，在交易訊號0以上，停損線以下
          #持有X長部位，Y空部位
          close_x = price_x['收盤價(元)'][next_date] #隔日開盤價
          close_y = price_y['收盤價(元)'][next_date]
          equity = fund + number_x * close_x * (1 - commision - tax - slip) - number_y * close_y * (1 + commision + slip) + interest_income #以明日收盤價計算明日權益
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[next_date] = equity #將明日權益數放入權益數序列
          cum_return[next_date] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[next_date], buy[next_date], offset[next_date] = None, None, None #不設定
        elif position == 0 and signal[i] == 1: #若沒部位，在交易訊號下界以下，停損線以上
          #賣X買Y
          position = -1 #設定為X空部位，Y長部位
          open_x = price_x['開盤價(元)'][next_date] #隔日開盤價
          open_y = price_y['開盤價(元)'][next_date]
          close_x = price_x['收盤價(元)'][next_date] #隔日收盤價
          close_y = price_y['收盤價(元)'][next_date]
          number_y = (fund / (1 + beta)) // (open_y * (1 + commision + slip)) #買入幾股Y
          number_x = (((fund * beta / (1 + beta)) // (open_x * (1 + commision + tax + short_cost + slip)))) // 1000 * 1000 #放空幾張X
          fund = fund - number_y * open_y * (1 + commision + slip) + number_x * open_x * (1 - commision - tax - short_cost - slip) #明日剩餘資金
          interest_income = open_x * 0.9 * short_interest / 365 #融券保證金利息收入
          equity = fund + number_y * close_y * (1 - commision - tax - slip) - number_x * close_x * (1 + commision + slip) + interest_income #以明日收盤價計算明日權益
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[next_date] = equity #將明日權益數放入權益數序列
          cum_return[next_date] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[next_date], buy[next_date], offset[next_date] = None, spread[next_date], None #設定買入點
        elif position == -1 and (signal[i] == 1 or signal[i] == 2): #若有X空部位，Y長部位，在交易訊號0以上，停損線以下
          #持有X空部位，Y長部位
          close_x = price_x['收盤價(元)'][next_date] #隔日收盤價
          close_y = price_y['收盤價(元)'][next_date]
          equity = fund + number_y * close_y * (1 - commision - tax - slip) - number_x * close_x * (1 + commision + slip) + interest_income #以明日收盤價計算明日權益
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[next_date] = equity #將明日權益數放入權益數序列
          cum_return[next_date] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[next_date], buy[next_date], offset[next_date] = None, None, None #不設定
        elif position == 0 and (signal[i] == 3 or signal[i] == 2 or signal[i] == 0 or signal[i] == 5): #若沒部位，在交易訊號上下界內，停損區間外
          #持有現金
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[next_date] = equity #將明日權益數放入權益數序列
          cum_return[next_date] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[next_date], buy[next_date], offset[next_date] = None, None, None #不設定
        elif position == 1 and (signal[i] == 2 or signal[i] == 1 or signal[i] == 5): #若有X長部位，Y空部位，在交易訊號0以下，停損線以上
          #平倉：賣X買Y
          open_x = price_x['開盤價(元)'][next_date] #隔日開盤價
          open_y = price_y['開盤價(元)'][next_date]
          equity = fund + number_x * open_x * (1 - commision - tax - slip) - number_y * open_y * (1 + commision + slip) + interest_income #以明日開盤價計算明日權益
          fund = equity #將權益數全部變為資金
          position, number_x, number_y, interest_income = 0, 0, 0, 0 #初始化部位、持有X、Y股票數量和融券利息收入為0
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[next_date] = equity #將明日權益數放入權益數序列
          cum_return[next_date] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[next_date], buy[next_date], offset[next_date] = None, None, spread[next_date] #設定平倉點
        elif position == -1 and (signal[i] == 3 or signal[i] == 4 or signal[i] == 0): #若有X空部位，Y長部位，在交易訊號0以上，停損線以下
          #平倉：買X賣Y
          open_x = price_x['開盤價(元)'][next_date] #隔日開盤價
          open_y = price_y['開盤價(元)'][next_date]
          equity = fund + number_y * open_y * (1 - commision - tax - slip) - number_x * open_x * (1 + commision + slip) + interest_income #以明日開盤價計算明日權益
          fund = equity #將權益數全部變為資金
          position, number_x, number_y, interest_income = 0, 0, 0, 0 #初始化部位、持有X、Y股票數量和融券利息收入為0
          ret = equity / equitys[today] - 1 #計算明日報酬
          equitys[next_date] = equity #將明日權益數放入權益數序列
          cum_return[next_date] = cum_return[today] * (1 + ret) #計算明日累積報酬
          sell[next_date], buy[next_date], offset[next_date] = None, None, spread[next_date] #設定平倉點
    return cum_return, equitys, buy, sell, offset #回傳累積報酬序列、權益數序列、買入點序列、賣出點序列和平倉點序列
  def perform(cum_return, equitys, spread, buy, sell, offset, year, year_test): #績效函數
    spread.plot(label='spread') #畫出殘差圖
    buy.plot(label='buy', style='o', markersize=5) #畫出買入點
    sell.plot(label='sell', style='o', markersize=5) #畫出賣出點
    offset.plot(label='offset', style='o', markersize=5) #畫出平倉點
    plt.legend() #顯示圖例
    plt.ylabel('spread') #坐標軸標籤
    plt.xlabel('Date')
    plt.title('spread & signal', fontsize=16) #圖表標題
    plt.show()
    equitys.plot() #畫出權益圖
    plt.axvline(x='2023-01-03', color='black') #垂直線分割樣本
    plt.show()
    #全部樣本
    cum_return_total = cum_return
    cum_return_total = (cum_return_total - 1) * 100 #將累積報酬改成百分比
    MDD_series = cum_return_total.cummax() - cum_return_total #計算最大回徹序列
    high_index = cum_return_total[cum_return_total.cummax() == cum_return_total].index #計算累積報酬最高點序列
    fig, ax = plt.subplots(figsize=(16,6))
    cum_return_total.plot(label='Cumulative Return', ax=ax, c='r') #畫出累積報酬率序列
    plt.fill_between(MDD_series.index, -MDD_series, 0, facecolor='r', label='DD') #畫出最大回徹序列
    plt.scatter(high_index, cum_return_total.loc[high_index], c='#02ff0f', label='High') #畫出累積報酬最高點序列
    plt.axvline(x=test_start, color='blue', linestyle='--') #垂直線分割樣本
    plt.legend()
    plt.ylabel('Return%')
    plt.xlabel('Date')
    plt.title('Return & MDD', fontsize=16)
    plt.show()
    #樣本內
    cum_return_in = cum_return[start:end] #取出樣本內累積報酬率序列
    daily_return = cum_return_in.diff(1) #計算單日報酬
    annual_return = cum_return_in.iloc[-1] ** (1 / year) - 1 #計算年化報酬
    annual_std = daily_return.std() * (240 ** 0.5) #計算年化標準差
    print(f'年化報酬率: {round(annual_return * 100, 2)}%') #年化報酬(%)
    print(f'累積報酬率: {round((cum_return_in.iloc[-1] - 1) * 100, 2)}%') #累積報酬(%)
    print(f'年化波動度: {round(annual_std * 100, 2)}%') 
    print(f'年化夏普值: {round(annual_return / annual_std, 2)}')
    #最大回徹圖
    cum_return_in = (cum_return_in - 1) * 100 #將累積報酬改成百分比
    MDD_series = cum_return_in.cummax() - cum_return_in #計算最大回徹序列
    MDD = round(MDD_series.max(), 2) #計算最大回徹
    print(f'MDD(%): {MDD}%')
    print(f'風報比: {round(cum_return_in.iloc[-1] / MDD, 2)}')
    print(f'{round(annual_return * 100, 2)}%', f'{round(annual_return / annual_std, 2)}', f'{MDD}%', f'{round(cum_return_in.iloc[-1] / MDD, 2)}')
    high_index = cum_return_in[cum_return_in.cummax() == cum_return_in].index #計算累積報酬最高點序列
    fig, ax = plt.subplots(figsize=(16,6))
    cum_return_in.plot(label='Cumulative Return', ax=ax, c='r') #畫出累積報酬率序列
    plt.fill_between(MDD_series.index, -MDD_series, 0, facecolor='r', label='DD') #畫出最大回徹序列
    plt.scatter(high_index, cum_return_in.loc[high_index], c='#02ff0f', label='High') #畫出累積報酬最高點序列
    plt.legend()
    plt.ylabel('Return%')
    plt.xlabel('Date')
    plt.title('Return & MDD', fontsize=16)
    plt.show()
    #樣本外
    cum_return_out = cum_return[test_start:test_end] #取出樣本外累積報酬率序列
    cum_return_out = cum_return_out / cum_return_out[test_start] #以樣本外第1天作為累積報酬率基準
    daily_return = cum_return_out.diff(1) #計算單日報酬
    annual_return = cum_return_out.iloc[-1] ** (1 / year_test) - 1 #計算年化報酬
    annual_std = daily_return.std() * ((240 * year_test) ** 0.5) #計算年化標準差
    print(f'年化報酬率: {round(annual_return * 100, 2)}%') #年化報酬(%)
    print(f'累積報酬率: {round((cum_return_out.iloc[-1] - 1) * 100, 2)}%') #累積報酬(%)
    print(f'年化波動度: {round(annual_std * 100, 2)}%')
    print(f'年化夏普值: {round(annual_return / annual_std, 2)}')
    #最大回徹圖
    cum_return_out = (cum_return_out - 1) * 100 #將累積報酬改成百分比
    MDD_series = cum_return_out.cummax() - cum_return_out #計算最大回徹序列
    MDD = round(MDD_series.max(),2) #計算最大回徹
    print(f'MDD(%): {MDD}%')
    print(f'風報比: {round(cum_return_out.iloc[-1] / MDD, 2)}')
    print(f'{round(annual_return * 100, 2)}%', f'{round(annual_return / annual_std, 2)}', f'{MDD}%', f'{round(cum_return_out.iloc[-1] / MDD, 2)}')
    high_index = cum_return_out[cum_return_out.cummax() == cum_return_out].index #計算累積報酬最高點序列
    fig, ax = plt.subplots(figsize=(16,6))
    cum_return_out.plot(label='Cumulative Return', ax=ax, c='r') #畫出累積報酬率序列
    plt.fill_between(MDD_series.index, -MDD_series, 0, facecolor='r', label='DD') #畫出最大回徹序列
    plt.scatter(high_index, cum_return_out.loc[high_index], c='#02ff0f', label='High') #畫出累積報酬最高點
    plt.legend()
    plt.ylabel('Return%')
    plt.xlabel('Date')
    plt.title('Return & MDD', fontsize=16)
    plt.show()

price_x = pd.read_csv("stock_price/9921.csv", index_col='年月日') #讀取 中華 股價 2204
price_x.index = pd.to_datetime(price_x.index)
price_y = pd.read_csv("stock_price/9914.csv", index_col='年月日') #讀取 和泰車 股價 2207
price_y.index = pd.to_datetime(price_y.index)
start = '2019-01-02' #形成期起始日期
end = '2022-12-30' #形成期結束日期
test_start = '2023-01-03' #交易期起始日期
test_end = '2023-12-29' #交易期結束日期
multiplier = 1.5 #交易訊號上下界是幾倍標準差
stop = 2 #停損訊號上下界是幾倍標準差
year = 4 #樣本內區間(年)，用來計算年化報酬
year_test = 1 #樣本外區間(年)，用來計算年化報酬

#做共整合檢定
alpha, beta = pair_trading.cointeration(price_x['收盤價(元)'], price_y['收盤價(元)'], start, end)

#建構配對交易策略
signal, spread = pair_trading.trade_signal(price_x['收盤價(元)'], price_y['收盤價(元)'], alpha, beta, start, end, test_start, test_end, multiplier, stop)

#策略回測
fund = 10000000
cum_return, equitys, buy, sell, offset = pair_trading.trade_account(price_x, price_y, signal, beta, spread, start, end, test_start, test_end, fund)

#回測績效績效
pair_trading.perform(cum_return, equitys, spread, buy, sell, offset, year, year_test)