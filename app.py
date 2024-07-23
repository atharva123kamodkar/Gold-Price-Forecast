# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt

# # Load the pickled model
# with open('final_model.pkl', 'rb') as file:
#     final_model = pickle.load(file)

# # Streamlit app title
# st.title('ARIMA Model Deployment')

# # Sidebar for user input
# st.sidebar.header('User Input')
# input_data = st.sidebar.text_area('Enter data for prediction (comma-separated values)', '')

# # Convert input data to DataFrame
# if input_data:
#     input_values = [float(i) for i in input_data.split(',')]
#     input_df = pd.DataFrame(input_values, columns=['input'])
    
#     # Predict using the loaded model
#     forecast = final_model.forecast(steps=len(input_df))
    
#     # Display the prediction
#     st.subheader('Prediction')
#     st.write(forecast)
    
#     # Plotting the prediction
#     st.subheader('Prediction Plot')
#     fig, ax = plt.subplots()
#     ax.plot(input_df.index, input_values, label='Input Data')
#     ax.plot(input_df.index, forecast, label='Prediction', linestyle='--')
#     ax.legend()
#     st.pyplot(fig)

# # Note: In case the prediction uses another method or requires additional steps, adjust accordingly.



#Flask app


from flask import Flask,render_template,request
import pickle
import pandas as pd

model=pickle.load(open('final_model.pkl','rb'))
pred=pickle.load(open('final_pred.pkl','rb'))

app=Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_days = 0
    pred = None
    index_future_dates = None

    if request.method == 'POST':
        # Get the number of days from the form
        forecast_days = int(request.form.get('days', 0))

        if forecast_days > 0:
            # Compute the future dates
            index_future_dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_days)

            # Assuming you have a DataFrame 'df' used for prediction
            df = pd.DataFrame()  # Replace with your actual DataFrame

            # Make prediction
            pred = model.predict(start=len(df), end=len(df) + forecast_days - 1, typ='levels').rename('Predicted_Price')
            pred.index = index_future_dates
            pred = pred.round(2)

    return render_template('index.html', forecast_days=forecast_days, index_future_dates=index_future_dates, pred=pred)


@app.route('/about')
def about_page():
    return render_template('about.html')

if __name__=='__main__':
    app.run(debug=True)

