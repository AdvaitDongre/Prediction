from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the data
df = pd.read_csv('ward_level_collated.csv')

# Create labels based on conditions (you can adjust this based on your use case)
df['Label'] = df.apply(
    lambda row: 'Reduce' if (row['Pop_Density_Category'] == 'Low' and row['Bus_Stop_Category'] == 'High') else
    'Increase' if (row['Pop_Density_Category'] == 'High' and row['Bus_Stop_Category'] == 'Low') else
    'None', axis=1
)

# Select features and target variable
X = df[['TOT_P_DEN', 'Bus_Stop_Count']]
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Define the prediction function
def predict(latitude, longitude):
    # Perform prediction using the loaded model
    prediction = clf.predict([[latitude, longitude]])[0]

    # Count the number of wards to reduce and increase bus stops
    reduce_bus_stops = df[df['Label'] == 'Reduce']['Ward_Names'].tolist()
    increase_bus_stops = df[df['Label'] == 'Increase']['Ward_Names'].tolist()

    return {'reduce': reduce_bus_stops, 'increase': increase_bus_stops, 'prediction': prediction}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    # Redirect to the analysis_results page
    return redirect(url_for('analysis_results'))


@app.route('/predict')
def handle_predict():
    # Get latitude and longitude from the query parameters
    latitude = float(request.args.get('latitude', 0))
    longitude = float(request.args.get('longitude', 0))

    # Perform prediction
    predictions = predict(latitude, longitude)

    # Check if the current location is in the reduce or increase list
    current_location = f"{latitude}, {longitude}"
    if current_location in predictions['reduce']:
        result = "hi"
    elif current_location in predictions['increase']:
        result = "no"
    else:
        result = "the number is appropriate"

    # Send predictions, chart URL, and result as JSON response
    return jsonify({'result': result})

# Flask route to render the analysis results page
@app.route('/analysis_results')
def analysis_results():
    # Perform necessary calculations and generate plots here
    # For demonstration purposes, we'll create a simple bar plot
    plt.figure(figsize=(8, 6))
    df['Label'].value_counts().plot(kind='bar', color=['red', 'green', 'blue'])
    plt.title('Distribution of Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()

    # Determine the recommendation based on the analysis
    recommendation = "Increase or Reduce based on analysis"

    # Render the analysis_results.html template with the generated chart and recommendation
    return render_template('analysis_results.html', chart_url=chart_url, recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
