import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import os

# Boolean Function Transformer
def yes_no_to_int(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0].astype(str)
    else:
        x = pd.Series(x.ravel()).astype(str)

    return (x.str.strip().str.lower().isin(["yes", "1", "true"])).astype(int).values.reshape(-1, 1)

# Combine City & District Transformer
def combineCityDistrict(df):
    df["City_District"] = df["City"].str.strip() + "_" + df["District"].str.strip()
    df.drop(['City', 'District'],  axis=1, inplace=True )
    return df

app = Flask(__name__)

# Global variable to hold the model
model = None

def load_model():
    global model
    if model is None:
        model_path = "model_pipeline2.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file '{model_path}' not found")
    return model

# Endpoint for predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Load model when needed
        current_model = load_model()
        
        data = request.get_json()
        if isinstance(data, dict):
            data = [data]
        
        df = pd.DataFrame(data)

        predict = current_model.predict(df)

        if hasattr(current_model.named_steps["regressor"], "estimators_"):
            # Prediksi dari semua pohon di RandomForest
            all_preds = np.stack([
                tree.predict(current_model.named_steps["preprocessor"].transform(df))
                for tree in current_model.named_steps["regressor"].estimators_
            ])

            mean_preds = all_preds.mean(axis=0)
            stdDev = all_preds.std(axis=0)

            # Confidence dalam persen (0-100)
            confidence = np.exp(-stdDev / (np.abs(mean_preds) + 1e-6)) * 100
            confidence = confidence.tolist()
        
        else:
            confidence = [None] * len(predict)
        
        predict = [round(float(p), 2) for p in predict]
        confidence = [round(float(c), 2) if c is not None else None for c in confidence]

        return jsonify({
            "status": "success",
            "prediction": predict,
            "confidence": confidence
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        load_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": True
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5375, debug=True)