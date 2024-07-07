import streamlit as st
import pandas as pd
import joblib
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import MinMaxScaler

# Attempt to load the model file using joblib
try:
    mlr_model = joblib.load('linear_regression_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file 'linear_regression_pipeline.pkl' not found. Please check the file path.")
    st.stop()

def preprocess_data(df):
    # Define columns to consider for preprocessing
    columns_to_preprocess = ['Water Temperature', 'Air Temperature', 'Barometric Pressure', 
                             'Adjusted Residuals', 'Wind Direction', 'Wind Gust']
    
    # Calculate mean for each relevant column
    mean_values = df[columns_to_preprocess].mean()
    
    # Handle NULL values (NaN) by replacing with mean
    df.fillna(mean_values, inplace=True)
    
    # Handle specific placeholder values like -9999 by replacing with mean
    df.replace(-9999, mean_values, inplace=True)
    
    return df

def predict_new_column(df, model, scaler, feature_columns):
    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    df['Predicted Sea Level'] = y_pred / 1000 
    
    return df

def main():
    st.header('Coral Sea, Sea Level Prediction', divider='blue')

    input_file = st.file_uploader(f'Upload csv file', type='csv')
    
    if input_file:
        tab1, tab2 = st.tabs(['Data Overview', 'Sea Level Prediction'])
        df = pd.read_csv(input_file)
        
        with tab1:
            pr = ProfileReport(df, explorative=True)
            st_profile_report(pr)
            
        with tab2:
        
            try:
                df_processed = preprocess_data(df)
            except KeyError as e:
                st.error(f"Unexpected column '{e.args[0]}' found in uploaded file. Please check the data format.")
                st.stop()

            try:
                scaler = mlr_model.named_steps['scaler']
                feature_columns = ['Water Temperature', 'Air Temperature', 'Barometric Pressure', 
                                   'Adjusted Residuals', 'Wind Direction', 'Wind Gust']
                predicted_df = predict_new_column(df_processed, mlr_model, scaler, feature_columns)
                
                st.subheader('Predicted Data')
                st.write(predicted_df)

                st.subheader('Sea Level Prediction Plot')
                st.line_chart(predicted_df.set_index(' Date & UTC Time')['Predicted Sea Level'])
            
            except KeyError as e:
                st.error(f"Unexpected column '{e.args[0]}' found in processed data. Please ensure data consistency.")
                st.stop()
            
if __name__ == '__main__':
    main()
