import numpy as np
import h5py
from plotly.tools import make_subplots
import streamlit as st
import plotly.graph_objects as go
import pages

st.set_page_config(page_title='PMRR PV+BESS+DG Dashboard', page_icon = 'battery', initial_sidebar_state = 'collapsed')
hide_streamlit_style = """
<style>

footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('PMRR Interactive Reporting Tool')
data_file = st.file_uploader('To begin, please upload your data file:',type='h5')
if data_file is not None:
    st.write(data_file.name)
    pages.PVBESSDG(data_file.name)
    

