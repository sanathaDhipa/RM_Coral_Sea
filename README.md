<!-- ABOUT THE PROJECT -->
## About The Project

### About

This project focuses on training Multiple Linear Regression and ANFIS (Adaptive Neuro-Fuzzy Inference System) models using hourly sea level and weather data from the [Australian Baseline Sea Level Monitoring Project](http://www.bom.gov.au/oceanography/projects/abslmp/data/index.shtml#table). The data comes from stations in the Coral Sea region, including Cape Ferguson, Rosslyn Bay, the Solomon Islands, and Vanuatu. Key features like Water Temperature, Air Temperature, Barometric Pressure, Adjusted Residuals, Wind Direction, and Wind Gust are used to predict sea level changes.

Data processing includes normalizing the features with MinMaxScaler and splitting the dataset into training and testing sets based on date. This ensures that the training and testing datasets are compatible with a specific batch size. The models are evaluated using metrics such as R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).

For deployment, we use Streamlit that accept a CSV file with feature data to get sea level predictions based on the trained models. Additionally, YData profiling is used to analyze the input CSV files, giving insights into the data's structure and quality.



### Built With

* [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
* [![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
* [![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
* [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
* [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
* [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)



<!-- GETTING STARTED -->
## Getting Started

### Run Locally

1. Clone the repo
2. Install necessary packages
   ```sh
   pip install -r requirements.txt
   ```
4. Run streamlit_app.py 
   ```js
   streamlit run streamlit_app.py
   ```



<!-- DEMO -->
## Demo

![Demo project](resources/demo_slvl.gif)
