import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.figure_factory as ff

# Load Data for Tables Here
arbitrage_data = np.load('data/arbitrage_data.npy')
wesm_rate = np.load('data/wesm_rates.npy')
discharge_rates = np.load('data/discharge_rates.npy')
echogen_battery_state = np.load('data/echogen_battery_state.npy')
etes_charge = np.load('data/echogen_charge.npy')
etes_discharge = np.load('data/echogen_discharge.npy')
cutoff_results = np.load('data/cutoff_results.npy')

# Margin Settings
left_margin = 20
right_margin = 20
top_margin = 50
bottom_margin = 10

def chart_echogen_arbitrage_summary():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=arbitrage_data[0],y=arbitrage_data[3],mode='lines+markers',name='Conservative'))
    fig.add_trace(go.Scatter(x=arbitrage_data[0],y=arbitrage_data[2],mode='lines+markers',name='Base Case'))
    fig.add_trace(go.Scatter(x=arbitrage_data[0],y=arbitrage_data[4],mode='lines+markers',name='Intermediate'))
    fig.add_trace(go.Scatter(x=arbitrage_data[0],y=arbitrage_data[1],mode='lines+markers',name='Aggressive'))
    fig.update_layout(
        title = 'Figure 1. Echogen Arbitrage Results',
        title_x = 0.5,
        xaxis_title = 'Storage Hours',
        yaxis_title = 'Income (USD)',
        legend = dict(title='WESM Case'),
        margin = dict(l=left_margin,r=right_margin,t=top_margin,b=bottom_margin),
    )
    return fig

def chart_echogen_selling_price():

    end_range=8760
    fig = go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0,end_range),y=wesm_rate,mode='lines',name='WESM Rate'))
    fig.add_trace(go.Scatter(x=np.arange(0,end_range),y=discharge_rates,mode='markers',name='WESM Rate during Discharge'))

    fig.update_layout(
        title='Figure 3. Selling Price during Discharging Hours - Intermediate Scenario',
        title_x = 0.5,
        xaxis_title='Hour of Year',
        yaxis_title='2020 WESM Rate (PhP/kWh)',
        margin = dict(l=left_margin,r=right_margin,t=top_margin,b=bottom_margin),
        #width = 1000,
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.0,
        xanchor="right",
        x=1
    ))
    fig.update_xaxes(range=[0,end_range])
    return fig

def makeHeatmap(data,title,suffix):

    heatmapdata = np.zeros(shape=(24,365))
    i = 0
    for x in range(0,365):
        for y in range(0,24):
            heatmapdata[y,x] = str(data[i])
            i = i+1

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z = heatmapdata,
        x = np.arange(1,365),
        y = np.arange(24),
        colorscale='balance',
        #showlegend=True,
        showscale=True,
        connectgaps=False,
        #smoothing = False,
        #my_colorsc=[[-1,'rgb(255,165,0'],[0,'rgb(255,0,0)'],[1,'0,0,255']],
        xgap = 0.1,
        ygap = 1,
        colorbar = dict(
        #title = 'Battery State',
        #tickmode = 'array',
        tickvals = [-1,-0.5,0,0.5,1],
        ticktext=['-1.0kW<br>(Discharging)','-0.5 kW','Idle','0.5 kW','1.0kW<br>(Charging)'],
        #ticks = 'outside',
        #tick0 = 0,
        dtick = 0.25,
    )
    ))

    fig.update_layout(
        title=suffix+title,
        title_x = 0.5,
        xaxis_title = 'Day of Year',
        xaxis_range=[1,50],
        yaxis_title = 'Hour of Day',
        legend_title = 'Battery State',
        margin = dict(l=left_margin,r=right_margin,t=top_margin,b=bottom_margin),
        #xmargin = go.layout.Margin(t=30,b = 30),
        #width = 1000,
        xaxis = dict(
            rangeslider = dict(visible=True)
        )
    )
    fig.update_yaxes(autorange="reversed",tick0=0, dtick = 2,)
    return fig

def chart_echogen_schedule():
    return makeHeatmap(echogen_battery_state,'Echogen Simulated Schedule for 2022','Figure 2. ')

def chart_echogen_cdc(efficiency):
    daily_charge = np.zeros(shape=365)
    daily_discharge = np.zeros(shape=365)
    daily_balance = np.zeros(shape=365)
    available_charge = np.zeros(shape=365)
    efficiency_check = np.zeros(shape=365)

    for i in range(0,365):
        if i == 0:
            start_index = 0
        else:
            start_index = (i*24)-2
        daily_charge[i] = np.sum(etes_charge[start_index:start_index+22])
        daily_discharge[i] = np.sum(etes_discharge[start_index:start_index+24])

        if i == 0:
            available_charge[i] = daily_charge[i]*efficiency
        else:
            available_charge[i] = (daily_charge[i]*efficiency) + daily_balance[i-1]
        
        daily_balance[i] = available_charge[i] - daily_discharge[i]

    fig = go.Figure()
    end_range = 366
    fig.add_trace(go.Bar(x=np.arange(1,end_range),y=daily_charge,name='Total Charge'))
    fig.add_trace(go.Bar(x=np.arange(1,end_range),y=available_charge,name='Available Charge'))
    fig.add_trace(go.Bar(x=np.arange(1,end_range),y=daily_discharge,name='Total Discharge'))
    fig.add_trace(go.Bar(x=np.arange(1,end_range),y=daily_balance,name='Battery Balance'))

    fig.update_layout(
        #barmode='relative',
        xaxis_title='Battery Cycle Number or Day of Year',
        xaxis_range=[0,10],
        yaxis_title = 'Energy (kW)',
        title = 'Figure 4. Echogen 2022 100MW/10h Battery Operation Schedule',
        title_x = 0.5,
        # legend = dict(
        #     orientation='h',
        #     yanchor = 'bottom',
        #     y = 0.95,
        #     xanchor='center',
        #     x = 0.5
        # ),
        xaxis = dict(
            rangeslider = dict(visible=True)
        ),
        margin = dict(l=left_margin,r=right_margin,t=top_margin,b=bottom_margin),

        )
    return fig

def table_2022_model():
    model2022 = {'Model':[1,2,3,4,5,6,7,8],
    'Description':['Echogen schedule using WESM rates for charging & discharging','Echogen schedule using BTM Energy Fee as charging rate','Echogen schedule using BTM Energy Fee and Capacity fee as charging rate','PMRR schedule using BTM Energy Fee as charging rate','PMRR schedule using BTM Energy Fee and Capacity fee as charging rate','PMRR schedule using WESM rates for charging & discharging','PMRR schedule using WESM rates for charging & discharging with minimum WESM rate during discharging','Echogen results from R. Wackerly'],
    'Model Code':['E-WESM','E-EF','E-EFCF','P-EF','P-EFCF','P-WESM-U','P-WESM-C','E-WESM-RW'],
    'Simulated by':['PMRR','PMRR','PMRR','PMRR','PMRR','PMRR','PMRR','Echogen-RW'],
    'Charge Rate':['WESM Rate','1.7 PhP/kWh','3.88 PhP/kWh','1.7 PhP/kWh','3.88 PhP/kWh','WESM','WESM Constrained','WESM']
    }
    table2022 = pd.DataFrame(model2022)
    fig = go.Figure(data=go.Table(
        header=dict(values=list(table2022.columns)),
        cells =dict(values=[table2022['Model'],table2022['Description'],table2022['Model Code'],table2022['Simulated by'],table2022['Charge Rate']]),
    ))
    fig.update_layout(
        title='Table 1. Models simulated with 2022 data',
        title_x = 0.5,
        margin = dict(l=left_margin,r=right_margin,t=top_margin,b=bottom_margin),
        height = 700
    )
    return fig

def chart_2022_results():
    results2022 = pd.read_csv('data/model2022.csv',header=0)
    fig = go.Figure()
    model2022codes = ['P-EF','E-EF','E-WESM','P-WESM-U','P-EFCF','E-EFCF']
    for i in range(0,len(model2022codes)):
        fig.add_trace(go.Scatter(
            x=results2022[results2022['Model Code']==model2022codes[i]].loc[:,'WESM Case'],
            y=results2022[results2022['Model Code']==model2022codes[i]].loc[:,'Profit PHP'],
            name = model2022codes[i], line=dict(width=1)
            ))
    
    cutoff_results = np.load('data/cutoff_results.npy')
    heatmapdata = np.zeros(shape=(11,4))
    print(len(cutoff_results))
    i=0
    for x in range(0,4):
        for y in range(0,11):
            heatmapdata[y,x] = round(cutoff_results[i,4],2)
            i=i+1
    x=['Conservative','Base Case','Intermediate','Aggressive']
    fig.add_trace(go.Scatter(x=x,y=[3707.73,4024.52,6465.53,6565.97],name='E-WESM-RW',line=dict(color='black',width=1)))
    y = []
    for i in range(0,11):
        string = 4.0+(i*0.2)
        y.append(str(string))
    for i in range(0,11):
        fig.add_trace(go.Scatter(x=x,y=heatmapdata[i],name=('P-WESM-C at: '+y[i]),line=dict(dash='dot',width=1)))
    fig.update_layout(
        xaxis_title='2022 Data Set',
        title = 'Figure 5. 2022 Model Data Results',
        title_x = 0.5,
        yaxis_title = 'Profit(PhP/kWh)',
        legend = dict(
            title='Model Code'
        ),
        height = 600
    )
    return fig

def chart_wesm_cutoff():
    cutoff_results = np.load('data/cutoff_results.npy')
    heatmapdata = np.zeros(shape=(11,4))
    print(len(cutoff_results))
    i=0
    for x in range(0,4):
        for y in range(0,11):
            heatmapdata[y,x] = round(cutoff_results[i,4],2)
            i=i+1
    x=['Conservative','Base Case','Intermediate','Aggresive']
    y = []
    for i in range(0,11):
        string = 4.0+(i*0.2)
        y.append(str(string))

    fig = go.Figure()
    for i in range(0,11):
        fig.add_trace(go.Scatter(x=x,y=heatmapdata[i],name=('Cutoff at: '+y[i]+' PhP/kWh'),line=dict(dash='dot',width=1)))
    fig.add_trace(go.Scatter(x=x,y=[2224.86,2503.09,6465.54,6564.74],name='Echogen (E-WESM)'))
    fig.add_trace(go.Scatter(x=x,y=[-555.1263,-278.2929,2236.776,2334.683],name='PMRR WESM Unconst. (P-WESM-U)'))
    fig.update_layout(
        title='Profit (PhP/kW) for various scenarios with 2022 data',
        title_x=0.5,
        yaxis_title='Profit (PhP/kW)',
        xaxis_title = '2022 Data Set',
    )
    # fig = ff.create_annotated_heatmap(heatmapdata,x=x,y=y,
    #     colorscale='RdBu',showscale=False)
    # fig.update_layout(
    #     title = 'Profit (PhP/kW) for WESM Charging with Cutoff',
    #     title_x = 0.5,
    #     yaxis_title='Cutoff Rate (PhP/kWh)',
    #     xaxis_title='2022 Dataset',
    #     legend_title='Profit (PhP)',
    # )
    # fig.update_xaxes(
    #     side = 'bottom' 
    # )
    # fig.update_yaxes(
    #     dtick = 0.2
    # )
    return fig

def table_LT_model():
    modelLT = pd.read_csv('data/modelLT.csv')
    fig = go.Figure(data=go.Table(
        header=dict(values=list(modelLT.columns[0:5])),
        cells =dict(values=[modelLT['Model'],modelLT['Model Description'],modelLT['Model Code'],modelLT['Charging Rate'],modelLT['Discharging Mode']]),
    ))
    fig.update_layout(
        title='Table 2. Models simulated with 2022-2045 data',
        title_x = 0.5,
        margin = dict(l=left_margin,r=right_margin,t=top_margin,b=bottom_margin),
        height = 700
    )
    return fig
    return modelLT.iloc[:,0:5]

def chart_LT_results():
    modelLT = pd.read_csv('data/modelLT.csv')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=modelLT['Model Code'],y=modelLT['Cost'],name='Charging Cost')
    )
    fig.add_trace(go.Bar(x=modelLT['Model Code'],y=modelLT['Income'],name='Income'))
    fig.add_trace(go.Bar(x=modelLT['Model Code'],y=modelLT['Profit'],name ='Profit'))
    fig.update_layout(
        title='Figure 6. Results for Long Term Simulations',
        title_x=0.5,
        xaxis_title='Model',
        yaxis_title='Value (PhP/kWh)',
        legend = dict(
            orientation='h',
            yanchor = 'bottom',
            y = 1,
            xanchor='center',
            x = 0.5
        ),
        margin = dict(l=left_margin,r=right_margin,t=top_margin,b=bottom_margin),
    )
    return fig