from decouple import config
MAIN_PATH =config('MAIN_PATH')

import base64
import numpy as np
import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET
from scipy import signal

from utils.sensors import apple_watch
from utils.sensors import balance_board
from utils import utils

from PIL import  Image
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
mpl.use("agg")
from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock  # Removed in newer matplotlib versions

# -- Set page config
apptitle = 'SensAnalytics DEMO'
st.set_page_config(page_title=apptitle, page_icon="🎛️", layout="wide") #:eyeglasses:
st.title('SensAnalytics')
utils.adjust_sidebar_width()
utils.hide_header_footer()

logo = Image.open('img/ohioh_logo.jpg')
st.sidebar.image(logo)
st.sidebar.markdown("# SensAnalytics")
st.sidebar.markdown("## About")
st.sidebar.markdown("This tool helps you visualise and analyse data captured from a variety of sensors. Use this sidebar menu to select sensors and set parameters. The visualisation and analysis will appear on the right. The tool supports both individual and cohort level analysis.")
st.sidebar.markdown("## Select Sensor & Parameters")
select_event = st.sidebar.selectbox('Sensor',['-- select a sensor from menue --', 'Balance Board (Bertec Force Plate)', 
                                                'Apple Watch 7', 'Fitbit Charge 4', 
                                                'Zephyr Bioharness 3', 'Samsung Galaxy Watch 4']
                                                ,help="The system includes integrations for the following sensors.")

if select_event == '-- select a sensor from menue --':
    st.subheader('Step 1: Select Sensor')
    st.subheader('Step 2: Select Analysis Type')
    st.subheader('Step 3: Select Data Input Method')
    st.subheader('Step 4: Preprocessing & Features')
    st.subheader('Step 5: Visualisations')
    st.subheader('Step 6: Analysis')

elif select_event == 'Balance Board (Bertec Force Plate)':
    st.subheader('Balance Board Analysis.')
    st.markdown("The balance board captures information regarding the postural sway.")
    analysis_type = st.sidebar.selectbox('Analysis Type', ['Individual Level', 'Cohort Level'])
    
    uploaded_files = None

    with st.sidebar.form(key='columns_in_form'):
        cohort_pd, file_names = [], []
        if analysis_type == 'Individual Level':
            data_input = st.selectbox('Data Input Method', ['CSV', 'Real Time'])
            if data_input == 'CSV':
                uploaded_files = st.file_uploader("Choose a CSV file", help="Upload appropriate CSV file for the sensor.")
                if uploaded_files is not None:
                    shows = pd.read_csv(uploaded_files)
                    cohort_pd.append(shows)
                    file_names.append(uploaded_files.name)
                else:
                    st.info(f"""👆 Upload a .csv file first.""")
            else:
                st.info('Please follow the following steps to pair your selected sensor.')
        else:
            uploaded_files = st.file_uploader("Choose a CSV file", help="Upload appropriate CSV file for the sensor.", 
                                                accept_multiple_files=True)
            if uploaded_files is not None:
                for uploaded_file in uploaded_files:
                    sub_data = pd.read_csv(uploaded_file)
                    cohort_pd.append(sub_data)
                    file_names.append(uploaded_file.name)
            else:
                st.info(f"""👆 Upload all .csv files first.""")
        sample_rate = st.selectbox('Sampling Frequency (Hz)', ['10', '20', '40', '100'])
        features = st.selectbox('Select Target Features', ['pathlength','area95', 'Example Scatter Plot','Histogram'])  # obtain feature types. 
        graph_type = st.selectbox('Visualisation Graph Type', ['Timeseries', 'Scatter Plot', 'Example Scatter Plot','Histogram'])  #'Scatter Plot Live'
        submitted = st.form_submit_button('Preprocess')
        
    
    graph_type = []
    preprocess_flag = False
    submit_analysis = False
    if submitted:
        preprocess_flag = False
        with st.spinner('Thank you for your patience. The data is being preprocessed and features calculated...'):
            processed_dfs, feature_arr = balance_board.preprocess(cohort_pd, lowpass_cutoff=20, lowpass_order=4, 
                                                                    decimate_order=sample_rate, demean=False, scale=1)
            # concat extracted features for the cohort 
            feature_df = []
            for r in range(0, len(file_names)):
                row = [file_names[r].split('.')[0]] + [i for i in feature_arr[r].values()]
                feature_df.append(row)
            column_names = ['subject'] + [i for i in feature_arr[0].keys()]
            feature_df=pd.DataFrame(feature_df, columns=column_names) 
            st.markdown('Data Preprocessing and Feature Extraction Complete!')
           
        if analysis_type == 'Individual Level':
            file_container = st.expander("Your uploaded csv file.")
            uploaded_files.seek(0)
            file_container.write(cohort_pd[0])
        elif analysis_type == 'Cohort Level':
            file_container = st.expander("Extracted feature summary for the cohort.")
            file_container.write(feature_df)
        preprocess_flag = True 

        fig = px.scatter(feature_df, x='pathlength', y='avg_velocity_ML', color="subject", labels={
                     "pathlength": "Pathlength",
                     "avg_velocity_ML": "Average Velocity Mediolateral",
                     "subject": "Participants"}, title="Example Scatter Plot")
        st.plotly_chart(fig)
        fig.show()

    # if preprocess_flag:
    #     with st.form(key='columns_in_form11'):        
    #         if preprocess_flag and (uploaded_files is not None):
    #             features = st.selectbox('Select Target Features', feature_arr[0].keys())  # obtain feature types. 
    #             graph_type = st.selectbox('Visualisation Graph Type', ['Timeseries', 'Scatter Plot', 'Example Scatter Plot','Histogram'])  #'Scatter Plot Live'
    #         submit_analysis = st.form_submit_button('Analysis')
    #         #ml_analysis = st.sidebar.selectbox('ML Analysis', ['Regression', 'Classification', 'Clustering'])

    # if submit_analysis and len(graph_type) != 0:
    #     shows = cohort_pd[0]
    #     if graph_type == 'Timeseries':
    #         fig = px.line(shows, x="Time", y="Fz", title="Sorted Input") 
    #         st.plotly_chart(fig)
    #         fig.show()

                # if g == 'Scatter Plot':
                #     CoPx = shows['CoPx'].to_numpy()
                #     CoPy = shows['CoPy'].to_numpy()
                #     CoPx = signal.resample(CoPx, 100) 
                #     CoPy = signal.resample(CoPy, 100)
                #     xmin = np.min(CoPx)
                #     xmax = np.max(CoPx)
                #     ymin = np.min(CoPy)
                #     ymax = np.max(CoPy)

                #     print('debug')
                #     print(CoPx[3])
                #     print(CoPx.shape)

    #             dataset = pd.DataFrame({'CoPx': CoPx, 'CoPy': CoPy}, columns=['CoPx', 'CoPy'])
                    
    #             # just plot
    #             fig = px.scatter(CoPx,CoPy)
    #             fig = px.scatter(dataset, 'CoPx', 'CoPy')
    #             fig.update_layout(yaxis_range=[ymin,ymax])
    #             fig.update_layout(xaxis_range=[xmin,xmax])
    #             st.plotly_chart(fig)
    #             #fig.show()

    #         if g == 'Example Scatter Plot':
    #             file_ = open("SwayAnimation_example.gif", "rb")
    #             contents = file_.read()
    #             data_url = base64.b64encode(contents).decode("utf-8")
    #             file_.close()

    #             st.markdown(
    #                 f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    #                 unsafe_allow_html=True,
    #             )
                    
                # if g == 'Scatter Plot Live':
                #     CoPx = shows['CoPx'].to_numpy()
                #     CoPy = shows['CoPy'].to_numpy()
                #     CoPx = signal.resample(CoPx, 10)
                #     CoPy = signal.resample(CoPy, 10)
                #     xmin = np.min(CoPx)
                #     xmax = np.max(CoPx)
                #     ymin = np.min(CoPy)
                #     ymax = np.max(CoPy)

                #     print('debug')
                #     print(CoPx[3])
                #     print(CoPx.shape)


                #     dataset = pd.DataFrame({'CoPx': CoPx, 'CoPy': CoPy}, columns=['CoPx', 'CoPy'])

                #     fig, ax = plt.subplots()
                #     ax.set_xlim([xmin, xmax])
                #     ax.set_ylim([ymin, ymax])


                    # scat = ax.scatter(CoPx[0], CoPy[0])
                    # def animate(i):
                    #     scat.set_offsets((CoPx[:i], CoPy[:i]))
                    #     return scat,

                    # print('debug copy')
                    # print(CoPx[:10])

                    # ani = animation.FuncAnimation(fig, animate, repeat=True, frames=len(CoPx) - 1, interval=50)

                    # # To save the animation using Pillow as a gif
                    # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                    # ani.save('scatter3.gif', writer=writer)
                    # print('done')

                    
                    # def animate(i):
                    #     df = dataset.iloc[:i]
                    #     cur = dataset.iloc[i]
                    #     ax.scatter(cur['CoPx'], cur['CoPy'], c='b')
                    #     ax.scatter(df['CoPx'], df['CoPy'], c='#ADD8E6')
                        

                    # ani = animation.FuncAnimation(fig, animate, repeat=True, frames=len(CoPx), interval=1)  # repeat=True, 
                    # #ani.save('scatter2.gif')  
                    # print('done')

                    # # To save the animation using Pillow as a gif
                    # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                    # ani.save('scatter7.gif', writer=writer)

elif select_event == 'Apple Watch 7':
    st.subheader('Apple Watch 7')
    st.markdown('The Apple Watch captured physiological signals such as the Heart Rate, Electrocardiogram (ECG), Oxygen Saturation.. etc')
    
    st.subheader('Electrocardiogram (ECG)')
    apple_watch_ecg = pd.read_csv(MAIN_PATH+'/apple_watch/ecg_sample.csv', skiprows=11)
    fig = px.line(apple_watch_ecg[0:1000]['Unit']) 
    st.plotly_chart(fig)

    st.markdown("To plot other metrics captured by your apple watch, please add the apple watch export xml file to the apple_watch data folder. Sample data is only provided for the ECG shown above.")

    FILE = MAIN_PATH+'/apple_watch/sample_export.xml'
    apple_data_extract = apple_watch.get_data(FILE)

    st.subheader('Heart Rate (InstantaneousBeatsPerMinute)')
    fig = px.line(apple_data_extract['apple_hr'], y="BPM", x="Time", title="") 
    st.plotly_chart(fig)

    st.subheader('Walking Steadiness')
    fig = px.line(apple_data_extract['walk_steadiness'], y="Walking Steadiness", x="Time", title="") 
    st.plotly_chart(fig)

    st.subheader('Walking Assymetry')
    fig = px.line(apple_data_extract['walk_assymetry'], y="Walk Assymetry", x="Time", title="") 
    st.plotly_chart(fig)

    st.subheader('Walking Double Support)')
    fig = px.line(apple_data_extract['double_support'], y="Walk Double Support", x="Time", title="") 
    st.plotly_chart(fig)

    st.subheader('Running Ground Contact')
    fig = px.line(apple_data_extract['ground_contact'], y="Running Ground Contact Time", x="Time", title="") 
    st.plotly_chart(fig)

    st.subheader('Running Vertical Oscillation)')
    fig = px.line(apple_data_extract['vertical_oscillation'], y="Running Vertical Oscillation", x="Time", title="") 
    st.plotly_chart(fig)

    st.subheader('Oxygen Saturation, Environment Sound, Barometric Pressure')
    st.markdown('Oxygen Saturation: ' + apple_data_extract['o2_sat'] +'%')
    st.markdown('Environment Sound: ' + apple_data_extract['sound'] + ' ' + apple_data_extract['sound_unit'])
    st.markdown('Barometric Pressure: ' + apple_data_extract['baro_pressure'])


elif select_event == 'Zephyr Bioharness 3':
    st.subheader('Zephyr Bioharness 3')
    st.markdown('The Zephyr Bioharness captured physiological signals such as the Heart Rate, Breath Rate, Activity (VMU) Accelertation (x,y,z), Posture (-180, 180, laying, standing etc), RtoR.')

    zephyr = pd.read_csv(MAIN_PATH+'/zephyr_bioharness3/sample_general.csv')
    zephyr = zephyr.drop(['Temp', 'Battery', 'ECGAmplitude', 'ECGNoise', 'BRAmplitude'], axis=1)
    
    zephyr = zephyr[100:-100]
    file_container = st.expander("Zephyr Bioharness3 General Data.")
    file_container.write(zephyr)

    st.subheader('Heart Rate')
    fig = px.line(zephyr, x="Timestamp", y="HR", title="") 
    st.plotly_chart(fig)

    st.subheader('Breath Rate')
    fig = px.line(zephyr, x="Timestamp", y="BR", title="") 
    st.plotly_chart(fig)

    st.subheader('Acceleration')
    fig = px.line(zephyr, x="Timestamp", y="Acceleration", title="") 
    st.plotly_chart(fig)

    st.subheader('Activity (Vector Magnitude Units (VMU))')
    fig = px.line(zephyr, x="Timestamp", y="Activity", title="") 
    st.plotly_chart(fig)
    
    # st.subheader('ECG')
    # fig = px.line(zephyr, x="Timestamp", y="ECGAmplitude", title="") 
    # st.plotly_chart(fig)

else:
    st.info('This sensor is not implemented, please select another sensor.')


