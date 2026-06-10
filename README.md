# Lightweight Anomaly Detection for Wearable Data Using Autoencoders

## Project Overview
This project detects anomalies in wearable health data using a lightweight autoencoder model. The system analyzes physiological and activity-related metrics such as heart rate, HRV, sleep, steps, respiratory rate, and energy expenditure to identify unusual patterns in user behavior.

## Objectives
- Detect personalized anomalies using wearable health data.
- Monitor circadian rhythm disruptions.
- Learn normal behavior using an unsupervised autoencoder.
- Identify anomalies through reconstruction error.
- Visualize results using an interactive dashboard.

## Features
- Wearable health data analysis
- Autoencoder-based anomaly detection
- Reconstruction error visualization
- Stress and heart rate monitoring
- Manual health input module
- Daily summary and insights dashboard

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow / Keras
- Plotly
- Streamlit

## Dataset
The dataset consists of Apple Health wearable data including:
- Heart Rate
- Heart Rate Variability (HRV)
- Step Count
- Sleep Duration
- Respiratory Rate
- Active Energy Burned
- Physical Effort
- Walking Metrics

## Methodology
1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Autoencoder Training
5. Reconstruction Error Calculation
6. Anomaly Detection
7. Dashboard Visualization

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <project-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run dashboard.py
```

## Results
- Detects abnormal behavioral patterns.
- Uses reconstruction error for anomaly detection.
- Generates personalized health insights.
- Provides interactive visualizations and summaries.

## Future Scope
- Real-time anomaly detection
- Multi-user support
- LSTM-based models
- Mobile application integration
- Cloud deployment

## Team Members

**Team D9**

- K. Keerthana (24251A05L9)
- M. Divya Sree (24251A05M9)
- P. Sai Priya Reddy (24251A05R3)

## Institution

G. Narayanamma Institute of Technology and Science (Autonomous)

## License
This project is developed for academic and educational purposes.