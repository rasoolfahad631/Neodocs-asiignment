import pandas as pd
import joblib


data = pd.read_excel("Python Dev Hemoglobin Data.xlsx")


calibration_model = joblib.load('hemoglobin_prediction_model.pkl')


def calibrate_phone_model(new_model_data):

    model_categories = ['model_' + str(cat) for cat in data['model'].unique()]
    
    
    new_model_encoded = pd.get_dummies(new_model_data['model']).reindex(columns=model_categories, fill_value=0)

    
    X_new_model = pd.concat([new_model_encoded, new_model_data.drop(columns=["model"])], axis=1)


    predicted_values = calibration_model.predict(X_new_model)

    return predicted_values

# Example usage: 
new_model_data = pd.DataFrame({
    'model': ['iPhone'],  
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
})

calibrated_values = calibrate_phone_model(new_model_data)
print("Calibrated hemoglobin values for the new model:", calibrated_values)
