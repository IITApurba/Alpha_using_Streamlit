import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import base64
import plotly.graph_objects as go
import plotly.express as px

# Streamlit page configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("Stock Analysis Dashboard")

# User inputs
ticker = st.text_input("Enter ticker symbol (e.g., AAPL):")
period = st.selectbox("Select period:", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])
interval = st.selectbox("Select interval:", ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'])

if st.button("Get Data"):
    if ticker and period and interval:
        data = yf.download(ticker, period=period, interval=interval)
        data.index = pd.to_datetime(data.index)

        # Candlestick Chart with Volume
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'],
                                             name='Candlestick')])
        fig.update_layout(title=f'{ticker} Candlestick Chart', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue', opacity=0.3, yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Volume'), height=700)
        st.plotly_chart(fig, use_container_width=True)

        # Technical Indicators
        data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        macd = ta.trend.MACD(data['Close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        bbands = ta.volatility.BollingerBands(data['Close'])
        data['bbands_upper'] = bbands.bollinger_hband()
        data['bbands_middle'] = bbands.bollinger_mavg()
        data['bbands_lower'] = bbands.bollinger_lband()
        data['dpo'] = ta.trend.DPOIndicator(data['Close']).dpo()
        dmi = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14)
        data['adx'] = dmi.adx()
        data['dmi_pos'] = dmi.adx_pos()
        data['dmi_neg'] = dmi.adx_neg()
        
        data['cci'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        data['roc'] = ta.momentum.ROCIndicator(data['Close']).roc()
        data['williamsr'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
        data['psar'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
        data['ema'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['sma'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()

        def CMO(data, period=14):
            diff = data.diff(1)
            gain = diff.where(diff > 0, 0)
            loss = -diff.where(diff < 0, 0)
            sum_gain = gain.rolling(window=period).sum()
            sum_loss = loss.rolling(window=period).sum()
            cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
            return cmo

        data['cmo'] = CMO(data['Close'])
        
        kc = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'])
        data['kc_upper'] = kc.keltner_channel_hband()
        data['kc_middle'] = kc.keltner_channel_mband()
        data['kc_lower'] = kc.keltner_channel_lband()
       
        vwap = ta.volume.VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
        data['vwap'] = vwap.vwap       
        
        # Manually calculate TEMA
        def TEMA(series, window):
            ema1 = series.ewm(span=window, adjust=False).mean()
            ema2 = ema1.ewm(span=window, adjust=False).mean()
            ema3 = ema2.ewm(span=window, adjust=False).mean()
            return 3 * (ema1 - ema2) + ema3

        # Calculate TEMA
        window = 30  # Example window period
        data['tema'] = TEMA(data['Close'], window)

        
        data['mfi'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
        data['fi'] = ta.volume.ForceIndexIndicator(data['Close'], data['Volume']).force_index()
        data['adi'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        data['eom'] = ta.volume.EaseOfMovementIndicator(data['High'], data['Low'], data['Close'], data['Volume']).ease_of_movement()
        data['dpo'] = ta.trend.DPOIndicator(data['Close']).dpo() 
 
        # RSI Indicator
        fig_rsi = px.line(data, x=data.index, y='rsi', title='RSI Indicator')
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD Indicator
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['macd'], name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name='Signal Line', line=dict(color='red')))
        fig_macd.add_trace(go.Bar(x=data.index, y=data['macd_diff'], name='MACD Diff', marker_color='green'))
        fig_macd.update_layout(title='MACD Indicator')
        st.plotly_chart(fig_macd, use_container_width=True)

        # Bollinger Bands
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['bbands_upper'], name='Bollinger High', line=dict(color='red', dash='dash')))
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['bbands_middle'], name='Bollinger Mid', line=dict(color='blue', dash='dash')))
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['bbands_lower'], name='Bollinger Low', line=dict(color='green', dash='dash')))
        fig_bb.update_layout(title='Bollinger Bands')
        st.plotly_chart(fig_bb, use_container_width=True)

        # Stochastic Oscillator
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=data.index, y=data['stoch_k'], name='%K', line=dict(color='blue')))
        fig_stoch.add_trace(go.Scatter(x=data.index, y=data['stoch_d'], name='%D', line=dict(color='red')))
        fig_stoch.update_layout(title='Stochastic Oscillator')
        st.plotly_chart(fig_stoch, use_container_width=True)

        # Detrended Price Oscillator (DPO)
        fig_dpo = px.line(data, x=data.index, y='dpo', title='Detrended Price Oscillator (DPO)')
        st.plotly_chart(fig_dpo, use_container_width=True)

        # Directional Movement Index (DMI)
        fig_dmi = go.Figure()
        fig_dmi.add_trace(go.Scatter(x=data.index, y=data['adx'], name='ADX', line=dict(color='blue')))
        fig_dmi.add_trace(go.Scatter(x=data.index, y=data['dmi_pos'], name='+DI', line=dict(color='green')))
        fig_dmi.add_trace(go.Scatter(x=data.index, y=data['dmi_neg'], name='-DI', line=dict(color='red')))
        fig_dmi.update_layout(title='Directional Movement Index (DMI)')
        st.plotly_chart(fig_dmi, use_container_width=True)

        # Download CSV
        csv = data.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="stock_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Calculate signals
        buy_counts = []
        sell_counts = []
        neutral_counts = []
        for index, row in data.iterrows():
            buy = 0
            sell = 0
            neutral = 0
            
            # RSI
            if row['rsi'] < 30:
                buy += 1
            elif row['rsi'] > 70:
                sell += 1
            else:
                neutral += 1
            
            # MACD
            if row['macd'] > row['macd_signal']:
                buy += 1
            elif row['macd'] < row['macd_signal']:
                sell += 1
            else:
                neutral += 1
            
            # Stochastic Oscillator
            if row['stoch_k'] < 20:
                buy += 1
            elif row['stoch_k'] > 80:
                sell += 1
            else:
                neutral += 1
            
            # ADX
            if row['adx'] > 25:
                neutral += 1  # ADX > 25 is a trend indicator, no direct buy/sell signal
            
            # CCI
            if row['cci'] < -100:
                buy += 1
            elif row['cci'] > 100:
                sell += 1
            else:
                neutral += 1
            
            # ROC
            if row['roc'] > 0:
                buy += 1
            elif row['roc'] < 0:
                sell += 1
            else:
                neutral += 1
            
            # Williams %R
            if row['williamsr'] < -80:
                buy += 1
            elif row['williamsr'] > -20:
                sell += 1
            else:
                neutral += 1
            
            # Bollinger Bands
            if row['Close'] < row['bbands_lower']:
                buy += 1
            elif row['Close'] > row['bbands_upper']:
                sell += 1
            else:
                neutral += 1
            
            # PSAR
            if row['Close'] > row['psar']:
                buy += 1
            elif row['Close'] < row['psar']:
                sell += 1
            else:
                neutral += 1
            
            # EMA
            if row['Close'] > row['ema']:
                buy += 1
            elif row['Close'] < row['ema']:
                sell += 1
            else:
                neutral += 1
            
            # SMA
            if row['Close'] > row['sma']:
                buy += 1
            elif row['Close'] < row['sma']:
                sell += 1
            else:
                neutral += 1
            
            # CMO
            if row['cmo'] < -50:
                buy += 1
            elif row['cmo'] > 50:
                sell += 1
            else:
                neutral += 1
            
            # Keltner Channel
            if row['Close'] < row['kc_lower']:
                buy += 1
            elif row['Close'] > row['kc_upper']:
                sell += 1
            else:
                neutral += 1
            
            # VWAP
            if row['Close'] > row['vwap']:
                buy += 1
            elif row['Close'] < row['vwap']:
                sell += 1
            else:
                neutral += 1
            
            # TEMA
            if row['Close'] > row['tema']:
                buy += 1
            elif row['Close'] < row['tema']:
                sell += 1
            else:
                neutral += 1
            
            # MFI
            if row['mfi'] < 20:
                buy += 1
            elif row['mfi'] > 80:
                sell += 1
            else:
                neutral += 1
            
            # Force Index
            if row['fi'] > 0:
                buy += 1
            elif row['fi'] < 0:
                sell += 1
            else:
                neutral += 1
            
            # Accumulation/Distribution Index
            if row['adi'] > 0:
                buy += 1
            elif row['adi'] < 0:
                sell += 1
            else:
                neutral += 1
            
            # On Balance Volume
            if row['obv'] > data['obv'].shift(1)[index]:
                buy += 1
            elif row['obv'] < data['obv'].shift(1)[index]:
                sell += 1
            else:
                neutral += 1
            
            # Ease of Movement
            if row['eom'] > 0:
                buy += 1
            elif row['eom'] < 0:
                sell += 1
            else:
                neutral += 1
            
            # Detrended Price Oscillator
            if row['dpo'] > 0:
                buy += 1
            elif row['dpo'] < 0:
                sell += 1
            else:
                neutral += 1
            
            # Directional Movement Index
            if row['dmi_pos'] > row['dmi_neg']:
                buy += 1
            elif row['dmi_pos'] < row['dmi_neg']:
                sell += 1
            else:
                neutral += 1
                 
            buy_counts.append(buy)
            sell_counts.append(sell)
            neutral_counts.append(neutral)
        
        results = pd.DataFrame({
            # 'datetime': data.index,
            'closing price': data['Close'],
            'Buy': buy_counts,
            'Sell': sell_counts,
            'Neutral': neutral_counts
        })

        st.subheader("Trading Signals")
        st.dataframe(results)
        
        b_last = results['Buy'].iloc[-1]
        s_last = results['Sell'].iloc[-1]
        n_last = results['Neutral'].iloc[-1]
        
        Indi_val = (b_last*1 + s_last*-1 +n_last*0)/(b_last+s_last+n_last)
        Indi_val = Indi_val*100

        fig_gauge_meter = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = Indi_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Bullishness", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "white", 'thickness': 0.1},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
            {'range': [-100, -60], 'color': "#1f77b4"},
            {'range': [-60, -20], 'color': "#aec7e8"},
            {'range': [-20, 20], 'color': "#ffbb78"},
            {'range': [20, 60], 'color': "#ff7f0e"},
            {'range': [60, 100], 'color': "#d62728"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95}}))

        fig_gauge_meter.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

        st.plotly_chart(fig_gauge_meter, use_container_width=True)

        st.markdown(
            """
            <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #f1f1f1;
                color: black;
                text-align: center;
                padding: 10px;
            }
            .footer a {
                color: #0a66c2; /* Color for links */
                text-decoration: none;
                margin: 0 10px;
            }
            .footer a:hover {
                text-decoration: underline;
            }
            </style>
            <div class="footer">
                <p>by Apurba</p>
                <p>
                    <a href="https://www.linkedin.com/in/apurba-kumar-show/" target="_blank">Instagram</a> |
                    <a href="https://github.com/IITApurba" target="_blank">GitHub</a> |
                    <a href="https://www.linkedin.com/in/apurba-kumar-show" target="_blank">LinkedIn</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.error("Please provide all inputs.")
