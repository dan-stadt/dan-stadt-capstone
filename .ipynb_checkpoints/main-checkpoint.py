import datetime
import os

import quandl
import numpy as np
import pandas as pd
import seaborn
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from pmdarima import auto_arima


class FieldName:
    def __init__(self, qdl_code, wb_code, title, is_commodity: bool, category):
        self.qdl_code = qdl_code
        self.wb_code = wb_code
        self.title = title
        self.is_commodity = is_commodity
        self.category = category
        self.conversion_rate = 0


class Fields:
    def __init__(self):
        # noinspection SpellCheckingInspection
        self.flist = [FieldName('RATEINF/CPI_USA', 'CPI', 'Consumer Price Index', False, None),
                      FieldName('ODA/POILWTI_USD', 'CRUDE_WTI', 'Crude Oil', True, 'Energy'),
                      FieldName('ODA/PNGASUS_USD', 'NGAS_US', 'Natural Gas', True, 'Energy'),
                      FieldName('ODA/PMAIZMT_USD', 'MAIZE', 'Corn / Maize', True, 'Food'),
                      FieldName('ODA/PWHEAMT_USD', 'WHEAT_US_HRW', 'Wheat', True, 'Food'),
                      FieldName('ODA/PBANSOP_USD', 'BANANA_US', 'Bananas', True, 'Food'),
                      FieldName('ODA/PPOULT_USD', 'CHICKEN', 'Poultry', True, 'Food'),
                      FieldName('ODA/PSUGAUSA_USD', 'SUGAR_US', 'Sugar', True, 'Food'),
                      FieldName('ODA/PCOPP_USD', 'COPPER', 'Copper', True, 'Metals'),
                      FieldName('ODA/PTIN_USD', 'Tin', 'Tin', True, 'Metals'),
                      FieldName('ODA/PZINC_USD', 'Zinc', 'Zinc', True, 'Metals')]
        self.qdl_codes = []
        self.wb_codes = []
        self.titles = []
        self.commodity_wbc = []
        self.energy_wbc = [self.flist[0].wb_code]
        self.energy_title = [self.flist[0].title]
        self.food_wbc = [self.flist[0].wb_code]
        self.food_title = [self.flist[0].title]
        self.metals_wbc = [self.flist[0].wb_code]
        self.metals_title = [self.flist[0].title]
        self.materials_wbc = [self.flist[0].wb_code]
        self.materials_title = [self.flist[0].title]
        for fld in self.flist:
            # Pull codes from fields
            self.qdl_codes.append(fld.qdl_code)
            self.wb_codes.append(fld.wb_code)
            self.titles.append(fld.title)
            if fld.is_commodity:
                self.commodity_wbc.append(fld.wb_code)
                # Sort fields into categories
                if fld.category == 'Food':
                    self.food_wbc.append(fld.wb_code)
                    self.food_title.append(fld.title)
                else:
                    self.materials_wbc.append(fld.wb_code)
                    self.materials_title.append(fld.title)
                    if fld.category == 'Energy':
                        self.energy_wbc.append(fld.wb_code)
                        self.energy_title.append(fld.title)
                    elif fld.category == 'Metals':
                        self.energy_wbc.append(fld.wb_code)
                        self.energy_title.append(fld.title)


fields = Fields()

# OIL: Earliest data available in 1982; Average crude price used prior to 1982
if __name__ == '__main__':
    # https://github.com/Nasdaq/data-link-python/blob/main/FOR_DEVELOPERS.md
    load_dotenv('keys.env')
    quandl.ApiConfig.api_key = os.environ.get('NASDAQ_DATA_LINK_API_KEY')
    df = pd.read_csv('cpi-and-commodity-data.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df_n_rows = df.shape[0]
    df_last_row = df_n_rows - 1
    df_last_date = df['DATE'][df_last_row]
    cpi_release = datetime.timedelta(days=45) + df_last_date
    # Check if a new CPI number has been released since the last column
    if datetime.datetime.now() >= cpi_release:
        start = df_last_date - datetime.timedelta(days=1)  # Get one day prior to latest date to normalize data
        end = datetime.date.today().isoformat()
        md = quandl.MergedDataset(fields.qdl_codes) \
            .data(params={'start_date': df_last_date, 'end_date': end, 'collapse': 'monthly'}).to_pandas()
        md.reset_index(inplace=True)
        df2 = pd.DataFrame()
        df2['DATE'] = md['Date']
        for field in fields.flist:
            wbc = field.wb_code
            qdc = field.qdl_code
            val1 = df[wbc][df_last_row]
            val2 = md[f'{qdc} - Value'][0]
            conv = val1 / val2
            df2[wbc] = md[f'{qdc} - Value'] * conv
        df2['DATE'] = md['Date']
        df2.drop(index=df2.index[0], inplace=True)
        df = pd.concat([df, df2], ignore_index=True)
        df.reset_index(drop=True, inplace=False).to_csv('test.csv')

    # Month over month percent change
    pct = pd.DataFrame(df)

    # Best fit line
    bfl = pd.DataFrame(df)

    # Figure 1. Line graphs for each commodity compared to CPI
    FIG1_ROWS, FIG1_COLS = 3, 4
    fig1, ax1 = plt.subplots(nrows=FIG1_ROWS, ncols=FIG1_COLS, num='Figure 1. Commodity Prices v. CPI',
                             layout='constrained')
    fig1.suptitle('Commodity Prices v. CPI')
    ax1_row, ax1_col = 0, 0
    ax1[FIG1_ROWS - 1][FIG1_COLS - 1].set_visible(False)
    ax1[FIG1_ROWS - 1][FIG1_COLS - 2].set_visible(False)
    colors = seaborn.color_palette('tab20', as_cmap=True)
    legend1 = []
    for val in range(4):
        legend1.append(Line2D([0], [0], color=colors.colors[val], lw=1))
    fig1.legend(legend1, ['CPI', 'CPI Trend Line', 'Commodity', 'Commodity Trend Line'], loc='lower right')
    print('Data processed.')

    # Normalize data to start commodity value at first CPI value
    for field in fields.flist:
        wbc, ttl = field.wb_code, field.title
        col = df[wbc]
        # https://www.mathworks.com/help/matlab/ref/polyfit.html
        if field.is_commodity:
            field.conversion_rate = df['CPI'][0] / col[0]
            df[wbc] = df[wbc] * field.conversion_rate
        x = np.linspace(start=0, stop=col.size, num=col.size)
        y = df[field.wb_code]
        p = np.polyfit(x, y, 1)
        bfl[wbc] = np.polyval(p, x)
        pct[wbc] = pd.DataFrame(df[wbc]).pct_change(1)
        plt_df = pd.DataFrame()
        if field.is_commodity:
            plt_df['DATE'] = df['DATE']
            plt_df['CPI'] = df['CPI']
            plt_df['CPI Best Fit Line'] = bfl['CPI']
            plt_df[f'{ttl}'] = df[wbc]
            plt_df[f'{ttl} Best Fit Line'] = bfl[wbc]
            plt_df = plt_df.melt('DATE', var_name='Legend', value_name='VALUE')
            seaborn.lineplot(data=plt_df, x='DATE', y='VALUE', hue='Legend', ax=ax1[ax1_row, ax1_col],
                             palette='tab20', legend=False) \
                .set(title=field.wb_code, xlabel='')
            ax1[ax1_row, ax1_col].set_ylabel('Index')
            ax1_col += 1
        if ax1_col == FIG1_COLS:
            ax1_row += 1
            ax1_col = 0

    # Figure 2 - Line graph of CPI and commodity price by category
    fig2, ax2 = plt.subplots(2, 1, layout='tight', num='Figure 2. Price Indices over Time')
    df.plot(x='DATE', y=fields.food_wbc, ax=ax2[0]).set(title='CPI and Raw Material Prices over Time', xlabel='Date')
    df.plot(x='DATE', y=fields.materials_wbc, ax=ax2[1]) \
        .set(title='CPI and Food Prices over Time', xlabel='Date')
    ax2[0].legend(fields.food_title)
    ax2[1].legend(fields.materials_title)

    # Figure 3- Histograms of monthly percent change
    fig3, ax3 = plt.subplots(nrows=FIG1_ROWS, ncols=FIG1_COLS, sharey='all',
                             num='Figure 3. Distribution of Monthly Percent Change in Price Indices')
    ax3_row, ax3_col = 0, 0
    br, bw = [-.025, .025], .002
    for field in fields.flist:
        if field.is_commodity:
            br, bw = [-.1, .1], .01
        if ax3_col == FIG1_COLS:
            ax3_col = 0
            ax3_row += 1
        seaborn.histplot(data=pct, x=field.wb_code, ax=ax3[ax3_row][ax3_col],  # kde=True,
                         binrange=br, binwidth=bw, common_bins=False) \
            .set(title=field.title, xlabel='')
        ax3[ax3_row][ax3_col].xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0, symbol='%'))
        ax3_col += 1
    plt.suptitle('Distribution of Monthly Percent Change in Price Indices')
    ax1[FIG1_ROWS - 1][FIG1_COLS - 1].set_visible(False)

    # Figure 4. CPI Forecasting
    fig4 = plt.figure(num='Figure 4. CPI Prediction vs. Actual')

    # https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    df_len = df.shape[0]
    TEST_LEN = 2
    train_len = df_len - TEST_LEN
    y = df['CPI']
    x = df[fields.commodity_wbc]
    train_y = df[['CPI']][:train_len]
    train_x = df[fields.commodity_wbc][: train_len]
    test_y = df[['DATE', 'CPI']][train_len: df_len]
    test_x = df[fields.commodity_wbc][train_len: df_len]
    print('Auto ARIMA started')
    # test_x = df[fields.commodity_wbc][train_len: df_len]
    ar = auto_arima(train_y, train_x, information_criterion='aicc', m=12, seasonal=True, test='adf',
                    start_p=0, start_q=0, max_p=3, max_q=3)  # enforce_stationarity=False
    # https://towardsdatascience.com/time-series-forecasting-with-arima-sarima-and-sarimax-ee61099e78f6
    print(ar.summary())
    ar.plot_diagnostics()
    print('Auto ARIMA processed.')
    test_y['TEST'] = ar.predict(X=test_x, n_periods=TEST_LEN)
    test_y.reset_index(inplace=True)
    print(test_y)
    test_y.plot(x='DATE', y=['CPI', 'TEST'])
    plt.show()
