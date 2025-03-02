import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Startup Success Prediction', layout='wide')

# Title
st.title('Startup Success Prediction Dashboard')

# File uploader
uploaded_file = st.file_uploader("Upload Startup Data (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.dataframe(df.head())
    
    # Ensure required columns exist
    required_columns = ['Funding Amount', 'Team Size', 'Years in Operation', 'Market Size', 'Success']
    if not all(col in df.columns for col in required_columns):
        st.error("Dataset must contain 'Funding Amount', 'Team Size', 'Years in Operation', 'Market Size', and 'Success'.")
    else:
        # Data preprocessing
        X = df[['Funding Amount', 'Team Size', 'Years in Operation', 'Market Size']]
        y = df['Success']  # 1 for successful, 0 for failed
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        with open("startup_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Accuracy score
        accuracy = model.score(X_test, y_test)
        st.write(f"### Model Accuracy: {accuracy:.2f}")
        
        # Prediction section
        st.sidebar.header("Predict Startup Success")
        funding = st.sidebar.number_input("Funding Amount ($M)", min_value=0.1, step=0.1)
        team_size = st.sidebar.number_input("Team Size", min_value=1, step=1)
        years = st.sidebar.number_input("Years in Operation", min_value=0, step=1)
        market_size = st.sidebar.number_input("Market Size ($B)", min_value=0.1, step=0.1)
        
        if st.sidebar.button("Predict Success"):
            new_data = pd.DataFrame([[funding, team_size, years, market_size]], 
                                    columns=['Funding Amount', 'Team Size', 'Years in Operation', 'Market Size'])
            prediction = model.predict(new_data)[0]
            result = "Success" if prediction == 1 else "Failure"
            st.sidebar.write(f"### Prediction: {result}")
        
        # Visualization
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(8, 5))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(plt)
