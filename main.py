# Import Libraries Required
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import charts

st.set_page_config(page_title='PMRR Reporting Dashboard', page_icon = 'battery', initial_sidebar_state = 'expanded')
st.sidebar.title('PMRR Reporting Dashboard')

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

password = st.sidebar.text_input(label='Enter password to access report')

if password ==st.secrets['report_password']:
    st.title('Echogen DER-VET Results Analysis')
    st.text('July 2021 | Prepared by D. Santos')
    st.markdown('This report explores the results presented by Echogen (R. Wackerly) for their 100MW/10H ETES. Results are then compared with internal models used to model Echogen ETES performance.')

    st.header('Executive Summary')
    st.markdown('Electrothermal Energy Storage (ETES) is an energy storage technology provided by Echogen. With the aim of deploying this technology in the Philippine market via Energy Arbitrage, the Echogen team has provided projected income for various storage configurations. The Echogen team used DER-VET, a simulation tool by EPRI, to simulate battery operation. These numbers were then verified and compared to results from using models developed in-house. An analysis of the results from all the simulations gives us the following insights:<ol><li> For a given storage configuration, there is a certain range of values for income independent of battery operation schedule.</li><li> The system<span>&#39;</span>s performance is more sensitive to the charging rate, where the most ideal scenario would be behind-the-meter charging at 1.7 PhP/kWh (Energy Fee only). </li><li>In-house models produce similar results as DER-VET for long-term simulations. In-house models also allow for better flexibility in simulating different scenarios specific to the Philippine Market.</li></ol>',unsafe_allow_html=True)


    st.header('The Echogen Model & Results')
    st.markdown('R. Wackerly used DER-VET by EPRI to simulate the ETES performance for a year. The market data was provided from WESM forecasts by R. Calano. Four models were used: Conservative, Base Case, Intermediate, Aggressive.')
    st.plotly_chart(charts.chart_echogen_arbitrage_summary())
    st.plotly_chart(charts.chart_echogen_schedule())
    st.markdown('R. Wackerly provided the sample data for a 100MW/10h system. The simulated schedule provided by DER-VET is illustrated in Figure 2. The ETES was charged during 22:00-6:00 of the next day. The amount of discharged power varied per day.')
    st.plotly_chart(charts.chart_echogen_selling_price())
    st.markdown('Analysis of the charging & discharging schedule computed by DER-VET showed that there are other factors involved in determining power dispatch. One trend observed was that the power was usually discharged when the WESM price for the hour was above 3.00 PhP/kWh as shown in Figure 3.')
    st.plotly_chart(charts.chart_echogen_cdc(0.63))
    st.markdown(' Total Charge and Total Discharge were taken from the data provided. Available Charge is computed as 63% (RTE) of the Total Charge. Battery Balance is the amount of power surplus/deficit for the given cycle. The operation schedule is illustrated in Figure 4.')
    st.markdown('With this given schedule, there were several instances where the power dispatched to the system was greater than the available stored energy. With the given data files we cannot confirm if there are other user-specified constraints that allow for these scenarios to happen. Internal attempts to run DER-VET also generated errors due to lack of technical parameters for auxiliary services. Simulations are also limited by the inputs that DER-VET allows which limits our ability to model the Philippine market accurately.')

    st.header('The PMRR Model')
    st.markdown('Internal models were created to simulate basic ETES performance when utilized for Energy Arbitrage. The model calculates the peformance accordingly: <ol><li>Parameters such as Battery capacity & Roundtrip Efficiency (RTE) were initialized</li> <li>The number of Charging Hours was specified (10 hours) and the corresponding allowed discharging hours (6.3 hours) was calculated based on RTE</li><li>The simulation ran for the selected year with an assumption of 0% charge at start.</li><li>For each charging/discharging cycle, the model looks for the hours with the ideal WESM rates within a 16-hour time frame (activity interval). It charges at the hours with the lowest rates. It discharges at the highest rates within the 16-hour interval after fully charged.</li><li>The battery cycles throughout the year and the equivalent cost, income, and profit is calculated.</li> </ol>',unsafe_allow_html=True)
    st.subheader('Behind-the-Meter Charging')
    st.markdown('The model was used to compute for ETES performance in a scenario where the facility was placed behind-the-meter of GNP plants. The charging costs was calculated given two scenarios:<ul><li>1.70 Php/kWh - Energy Fee only</li><li>3.88 Php/kWh - Energy + Capacity Fee</li></ul> Two modes of discharging were also considered: <ul><li>Adjacent - Power was discharged power within adjacent hours with the highest total WESM rates </li><li>Non-Adjacent - Power was discharged at hours with the highest WESM rate</li></ul>',unsafe_allow_html=True)
    st.subheader('Charging from WESM')
    st.markdown('For the simulation of Energy Arbitrage with charging from WESM, charging cost is computed as: <br/><center>$total\,cost = \sum charge(kW)* hourly \, WESM \, Rate(Php/kW)$</center><br/>Likewise, the total income is computed as:<br/><center>$total\,cost = \sum discharge(kW)* hourly \, WESM \, Rate(Php/kW)$</center><br/>The profit is calculated as:<br/><center>$Profit = Income - Cost$</center><br/>',unsafe_allow_html=True)
    st.markdown('The model results for 2022 data were simulated accordingly:<ul><li><b>Unconstrained</b> - Stored energy was discharged at the hours with the highest WESM rates.</li><li><b>Constrained</b> - Stored energy was discharged at the hours with the highest WESM rate and exceeded the minimum rate specified.</li>',unsafe_allow_html=True)
    # PMRR WESM Charging Explanation
    # PMRR Unconstrained
    # PMRR Constrained

    st.header('Model Comparison & Results Analysis')
    st.subheader('Comparison for 2022 Data')
    st.markdown('A summary of the models that used 2022 data for this analysis is presented below.All models were ran using the 4 variations of 2022 Data by R. Calano. The simulations were ran for a 10-hour system (10h charging/6.3h discharging)')
    #st.table(charts.table_2022_model().set_index('Model'))
    st.plotly_chart(charts.table_2022_model())
    st.plotly_chart(charts.chart_2022_results())

    st.markdown('Figure 5 shows the summary of results for various simulations to compared with the data from R. Wackerly (E-WESM). Insights from this are as follows: <ol><li>Simulation from Echogen follows the same algorithm as our WESM charging as verified by the close values for Intermediate for E-WESM-RW and E-WESM. The variance with other data sets may be due to a different operation schedule or the existence of a constraint (minimum rate for discharging).</li><li>Adding a constraint of minimum rate for discharging only increases the loss of profit in a system. It is observed that at a certain level (Around 4.0 Php/kWh) that there is very little difference in putting this constraint on the simulation.</li><li>At fixed charging rates it is observed that differences in battery operation are negligible using certain data sets. At lower charging rates (P-EF & E-EF), difference is negligible for Intermediate & Aggressive scenarios. At higher charging rates (P-EFCF & E-EFCF), difference is negligible for Conservative & Base Case Scenarios. If we were to consider the most realistic setup of having a behind-the-meter setup (EF+CF charging), ETES will still perform poorly for Y1 independent of the battery operation schedule.</li><li>It is observed that E-WESM-RW and P-WESM-U are basically the same results with E-WESM-RW being higher by around Php 4200-4300/kW. This is observed across all datasets. Higher results from E-WESM-RW may be attributed to the higher amount of discharged energy despite the improbable battery operation schedule.</li> </ol>',unsafe_allow_html=True)
    st.markdown('Given these insights, we can assume that the echogen model has a different approach to modelling results for Conservative and Base Case datasets.This may lead to an overestimation of projected profits without knowing the actual source of the added value. It is possible that other features of DER-VET optimizes the schedule to produce these results.')
    #st.plotly_chart(charts.chart_wesm_cutoff())
    # st.markdown('<table><thead><tr><th>Model Number</th><th>Trial</th></try></thead></table>',unsafe_allow_html=True)

    st.subheader('Long Term Simulations (2022-2045)')
    st.markdown('The following data are from long term simulations using WESM rates from 2022 to 2045 (23 Years). The data set used for 2022 was Intermediate to compare with the results given by R. Wackerly. For Echogen<span>&#39;</span>s data, their given battery operation schedule for 2022 Intermediate was used on 2023-2045 WESM data to obtain results.',unsafe_allow_html=True)
    st.plotly_chart(charts.table_LT_model())
    st.plotly_chart(charts.chart_LT_results())
    st.markdown('From these results, the following are observed:<ul><li>No matter what model is used in calculating, the revenue generated from discharging at WESM prices will always fall within a certain range (330-360k Php/kW). This comes from the fact that power will always be dispatched at ideal hours and only the volume of dispatched energy per hour will cause variation in the values.</li><li>Running on adjacent discharging allows for more cycles per year but does not translate to higher long term profits</li></ul>',unsafe_allow_html=True)

    st.header('Conclusion and Recommendations')
    st.markdown('Given the results of the analysis we can draw onto these conclusions: <ul><li>For a given configuration of ETES and a set of modeled WESM rates, battery operation schedule has little effect on the profit generated. Income will also fall at a certain range across different models.</li><li>The ETES is highly most feasible for behind-the-meter installation with a fixed charging fee at 1.7Php/kWh (Energy Fee only). For other scenarios, it is best to explore other income streams (auxiliary services) which are unfortunately unavailable in the Philippine Market</li></ul>In the event of further evaluation of the system, it is recommended that we proceed using our internal models. DER-VET, while very useful, is a black box with a lot of unknown configurations. This analysis has shown that we can attain similar values with more confidence using our internal models.',unsafe_allow_html=True)


    st.subheader('Appendix')
    with st.beta_expander('Definition of Terms'):
        st.markdown('1. **ETES** - *Electrothermal Energy Storage* - storage technology provided by Echogen')
        st.markdown('2. **DER-VET** - *Distributed Energy Resource Value Estimation Tool* - Simulation technology created by EPRI (Electric Power Research Institute)')
    with st.beta_expander('Echogen Simulation Assumptions'):
        st.markdown('1. *Foreign Exchange Rate*: 1 USD = 47.17 PHP')
        st.markdown('2. *Capital Cost*: $150,000,000')
        st.markdown('3. *Charge Rating*: 100,000 kW')
        st.markdown('4. *Discharge Rating*: 100,000 kW')
        st.markdown('5. *Duration*: 10 Hours')
        st.markdown('6. *Energy Rating*: 1,000,000 kWh')
        st.markdown('7. *Roundtrip Efficiency*: 63%')
        st.markdown('8. *Upper Limit on SOC*: 1%')
