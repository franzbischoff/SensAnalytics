import streamlit as st

def hide_header_footer():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.block-container {padding-top:1rem;}
            section[data-testid="stSidebar"] div.block-container {
                padding-top:0rem;
                top: -60px;
                # background-color: white;
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def adjust_sidebar_width():
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#import requests, os
# from gwpy.timeseries import TimeSeries
# from gwosc.locate import get_urls
# from gwosc import datasets
# from gwosc.api import fetch_event_json
#from copy import deepcopy

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data
# def load_gw(t0, detector, fs=4096):
#     strain = TimeSeries.fetch_open_data(detector, t0-14, t0+14, sample_rate = fs, cache=False)
#     return strain

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data
# def get_eventlist():
#     allevents = datasets.find_datasets(type='events')
#     eventset = set()
#     for ev in allevents:
#         name = fetch_event_json(ev)['events'][ev]['commonName']
#         if name[0:2] == 'GW':
#             eventset.add(name)
#     eventlist = list(eventset)
#     eventlist.sort()
#     return eventlist

# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################