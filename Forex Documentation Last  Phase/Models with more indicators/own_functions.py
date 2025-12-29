def label_data(df,mult1,mult2,loops,symbol,print_data=True):

    import warnings
    import pandas as pd
    warnings.filterwarnings('ignore')
    from pathlib import Path

    mean_candle=0
    for i in range(len(df)):
        mean_candle= mean_candle + (df['high'][i]-df['low'][i])
    mean_candle=mean_candle/len(df)
    print("Mean Candle:", mean_candle)
    
    b_sum=0
    s_sum=0
    b_cols = ['b_sum','mean_candle','b_sl', 'b_tp','b_lost','b_win','b_nothing','loops']
    b_data = pd.DataFrame(columns=b_cols)
    s_cols = ['s_sum','mean_candle','s_sl', 's_tp','s_lost','s_win','s_nothing','loops']
    s_data = pd.DataFrame(columns=s_cols)

    for m1 in range(len(mult1)):
        b_sl = mean_candle*mult1[m1]
        s_sl = mean_candle*mult1[m1]
        for m2 in range(len(mult2)):
            b_tp = mean_candle*mult2[m2]
            s_tp = mean_candle*mult2[m2]
            for i in range(len(df)-loops):

                price = df['open'][i]
                price_b_sl = df['open'][i]-b_sl
                price_b_tp = df['open'][i]+b_tp
                price_s_sl = df['open'][i]+s_sl
                price_s_tp = df['open'][i]-s_tp

                for j in range(loops):

                    if df['low'][i+j] < price_b_sl: 
                        df['b_flag'][i]=0
                        b_sum= b_sum - b_sl
                        break
                    elif df['high'][i+j] > price_b_tp: 
                        df['b_flag'][i]=1
                        b_sum= b_sum + b_tp
                        break
                    else: df['b_flag'][i]=0
                
                for j in range(loops):
                        
                    if df['high'][i+j] > price_s_sl: 
                        df['s_flag'][i]=0
                        s_sum= s_sum - s_sl
                        break
                    elif df['low'][i+j] < price_s_tp:
                        df['s_flag'][i]=1
                        s_sum= s_sum + s_tp
                        break
                    else: 
                        df['s_flag'][i]=0

            b_raw_data = {'b_sum': [b_sum],'b_sl': [b_sl],'b_tp': [b_tp],'b_lost':[df['b_flag'].value_counts()[0]],
                        'b_win':[df['b_flag'].value_counts()[1]],'b_nothing':[df['b_flag'].value_counts()[0]],
                        'mean_candle':[mean_candle],'loops':[loops]}
            b_temp = pd.DataFrame(b_raw_data, columns=b_cols)
            b_data=pd.concat([b_data,b_temp],ignore_index=True) 
            b_sum=0
            
            s_raw_data = {'s_sum': [s_sum],'s_sl': [s_sl],'s_tp': [s_tp],'s_lost':[df['s_flag'].value_counts()[0]],
                        's_win':[df['s_flag'].value_counts()[1]],'s_nothing':[df['s_flag'].value_counts()[0]],
                        'mean_candle':[mean_candle],'loops':[loops]}
            s_temp = pd.DataFrame(s_raw_data, columns=s_cols)
            s_data=pd.concat([s_data,s_temp],ignore_index=True)
            s_sum=0

    if print_data==True:
        print("Mean Candle:", mean_candle)
        print('\n')        
        print("b_data:", b_data)
        print('\n')
        print("s_data:", s_data)
        print('\n')
        print(df['b_flag'].value_counts())
        print('\n')
        print(df['s_flag'].value_counts())
        print('\n')     

        b_data=b_data.sort_values(by=['b_sum'],ascending=False)
        b_data.insert(4, 'Q', b_data['b_tp']/b_data['b_sl'])

        filepath_xls = Path(r'Data\\' + symbol + '_Buy.xlsx') 
        filepath_xls.parent.mkdir(parents=True, exist_ok=True) 
        b_data.to_excel(filepath_xls)
        print(b_data)

        s_data=s_data.sort_values(by=['s_sum'],ascending=False)
        s_data.insert(4, 'Q', s_data['s_tp']/s_data['s_sl'])
        filepath_xls = Path(r'Data\\' + symbol + '_Sell.xlsx') 
        filepath_xls.parent.mkdir(parents=True, exist_ok=True) 
        s_data.to_excel(filepath_xls)
        print(s_data)   
   
    


