import joblib
import pandas as pd 

model = joblib.load('hemoglobin_prediction_model.pkl')

# Here we  are predicting 'hb_value' for 'iPhone' model by provididng a sample input from our dataset
# For provided values hb_value in dataset foriPhone model is 14.1 and predicted value is 15.73
new_data = {
    'mid_b_mean': [50.67931987],
    'mid_g_mean': [31.97796119],
    'mid_r_mean': [173.8093081],
    'mid_b_median': [46],
    'mid_g_median': [32],
    'mid_r_median': [177],
    'mid_l_mean': [99.23359752],
    'mid_a_mean': [185.0770812],
    'mid_br_mean': [155.4250442],
    'mid_l_median': [99],
    'mid_a_median': [185],
    'mid_br_median': [158],
    'mid_h_mean': [175.8971356],
    'mid_s_mean': [209.3584312],
    'mid_v_mean': [179.2038019],
    'mid_h_median': [177],
    'mid_s_median': [209],
    'mid_v_median': [178],
    'model_IN2011': [0],
    'model_CPH2413': [0],
    'model_SM-A032F': [0],
    'model_moto g82 5G': [0],
    'model_motorola edge 40': [0],
    'model_Pixel 6 Pro': [0],
    'model_iPhone': [1],
    'model_I2203': [0],
    'model_vivo 1917': [0]
}

new_data_df = pd.DataFrame(new_data)

predicted_values = model.predict(new_data_df)

print("Predicted hemoglobin values:", predicted_values)