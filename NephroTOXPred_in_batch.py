import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import time
from PIL import Image

# Clear previous prediction results file to reduce storage usage
if os.path.exists("prediction_results.csv"):
    os.remove("prediction_results.csv")

@st.cache_data
def load_image(image_path):
    return Image.open(image_path)

def run_progress():
    progress_bar = st.empty()
    for i in range(10):
        progress_bar.progress(i / 10, 'Progress')
        time.sleep(0.5)
    with st.spinner('Loading...'):
        time.sleep(2)
    progress_bar.empty()

# Load and display the logo
logo = load_image("./logo.png")
st.image("./logo.png")

def get_fingerprints(smiles):
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        # Calculate MACCS fingerprints
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_bits = np.array(maccs_fp, dtype=int).tolist()

        # Calculate ECFP4 fingerprints
        ecfp4_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        ecfp4_bits = np.array(ecfp4_fp, dtype=int).tolist()

        return maccs_bits, ecfp4_bits
    except Exception as e:
        st.write("**Invalid SMILES string. Unable to perform subsequent calculations. **")
        return None, None

def generate_feature_vector(smiles, feature_order):
    maccs_bits, ecfp4_bits = get_fingerprints(smiles)
    if maccs_bits is None or ecfp4_bits is None:
        return None

    feature_vector = []
    for feature in feature_order:
        if feature.startswith("MACCS_"):
            index = int(feature.split("_")[1]) 
            feature_vector.append(maccs_bits[index])
        elif feature.startswith("ECFP4_bitvector"):
            index = int(feature.split("bitvector")[1])
            feature_vector.append(ecfp4_bits[index])

    return feature_vector

st.write("Supported by the service of Xiuqing Zhu at the AI-Drug Lab, the affiliated Brain Hospital, Guangzhou Medical University, China. If you have any questions, please feel free to contact me at 2018760376@gzhmu.edu.cn. ")

# Define feature names
feature_df = pd.read_csv('./features_for_ML.csv')
feature_names = feature_df['Features'].values.tolist()

# Load the model
model = joblib.load('./Model_final.joblib')

# Streamlit user interface
st.title("Nephrotoxic Component Predictor")

st.write("**Upload a CSV file with compound names and SMILES for batch prediction.**")

# Display sample CSV content
st.write("Example:")
st.write(pd.DataFrame({
    "Compound Name": ["Compound1", "Compound2", "Compound3"],
    "Smiles": ["CCO", "O=C=O", "C1=CC=CC=C1"]
}))

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

run_progress()

if uploaded_file is not None:
    try:
        # Ensure the uploaded file is CSV
        if not uploaded_file.name.endswith('.csv'):
            st.write("**File format error: Please upload a CSV file.**")
        else:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            # Ensure required columns exist
            if 'Compound Name' in df.columns and 'Smiles' in df.columns:
                results = []
                for index, row in df.iterrows():
                    smiles = row['Smiles']
                    compound_name = row['Compound Name']

                    # Generate feature vector
                    feature_vector = generate_feature_vector(smiles, feature_names)
                    
                    if feature_vector is None:
                        st.write(f"**Invalid SMILES for {compound_name}.**")
                        continue
                    
                    features = np.array([feature_vector])
                    
                    # Predict class and probabilities
                    predicted_class = model.predict(features)[0]
                    predicted_proba = model.predict_proba(features)[0]

                    # Generate advice based on prediction results
                    probability = round(predicted_proba[predicted_class] * 100, 2)

                    # Get important features
                    important_features = [feature_names[i] for i, value in enumerate(feature_vector) if value == 1]

                    # Append results
                    results.append({
                        "Compound Name": compound_name,
                        "Smiles": smiles,
                        "Class": predicted_class,
                        "Probability(%)": probability,
                        "Molecular Fingerprints": important_features
                    })

                # Create a DataFrame from results
                results_df = pd.DataFrame(results)

                # Display results
                st.write(results_df)

                # Save results to CSV
                results_csv = results_df.to_csv(index=False).encode('utf-8')

                # Download link
                st.download_button(
                    label="Download CSV",
                    data=results_csv,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )
            else:
                st.write("**The uploaded CSV file must have 'Compound Name' and 'Smiles' columns.**")

    except Exception as e:
        st.write("**Error processing the file. Please ensure it is a valid CSV file.**")

else:
    st.write("**No file uploaded yet.**")
