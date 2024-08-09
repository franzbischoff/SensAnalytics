![header](img/ohioh_logo.jpg)
<h1>SensAnalytics</h1>

<h2>About</h2>
Wearable sensors, among other informatics solutions, are readily accessible to enable noninvasive remote monitoring in healthcare. While providing a wealth of data, the wide variety of such sensing systems and the differing implementations of the same or similar sensors by different developers complicate comparisons of collected data. SensAnalytics is an tool that provides uniform methods. The use of the tool is demonstrated using a case study focussing on analysing balance data. Please read the associated paper for further details.


<h2>Installation</h2>

* Create and activate a Python3.9.2 virtual environment.<br>
* Clone the repository: <code>git clone git@github.com:chirathyh/SensAnalytics.git</code>.<br>
* Go to the project folder (SensAnalytics): <code>cd SensAnalytics</code><br>
* Install the required Python libraries <code>requirements.txt</code>. <br>
* Create an environment file <code>.env</code> at the root of the project (<code>echo "MAIN_PATH=$(pwd)">.env</code>) folder with <code>MAIN_PATH=path-to-this-project</code>.<br>
* Run the app <code>streamlit run app.py</code>. 

<h2>Notes</h2>

* Different sensors can be added to the tool, which requires the implementation of data extraction scripts, preprocessing pipelines (if applicabvle). Currently, a balance board, Zephyre Bioharness, and Apple Watch 7 sensors are demonstrated. <br>
* Minimal sample data is provided. You can add use your own data in the visualisations. <br>


<h2>Citation</h2>

```
@inproceedings{hettiarachchi2024sensanalytics,
  title={Optimising Personalised Medical Insights by Introducing a Scalable Health Informatics Application for Sensor Data Extraction, Preprocessing, and Analysis},
  author={C. Hettiarachchi, R. Vlieger, W. Ge,  D. Apthorpe, Daskalaki, A. Brustle, H. Suominen},
  booktitle={Health Innovation Community (HIC) Conference},
  year={2024},
}
```