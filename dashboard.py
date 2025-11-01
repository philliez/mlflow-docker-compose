import dash
from dash import dcc, html
import base64
import os

# --- Configuration ---
REPORT_FILE = 'analysis_report.txt'
CONFUSION_MATRIX_FILE = 'confusion_matrix.png'
SCORE_DIST_FILE = 'raw_score_distribution.png'

app = dash.Dash(__name__)

# --- Helper function to read image files ---
def encode_image(image_file):
    with open(image_file, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('ascii')
    return f'data:image/png;base64,{encoded_image}'

# --- Read report and image data ---
try:
    with open(REPORT_FILE, 'r') as f:
        report_text = f.read()
except FileNotFoundError:
    report_text = "Analysis report not found. Please run the pipeline first."

try:
    cm_image = encode_image(CONFUSION_MATRIX_FILE)
except FileNotFoundError:
    cm_image = None

try:
    score_dist_image = encode_image(SCORE_DIST_FILE)
except FileNotFoundError:
    score_dist_image = None

# --- Dashboard Layout ---
app.layout = html.Div([
    html.H1("BTCUSDT Win/Loss Prediction Dashboard"),

    html.Div([
        html.H2("Analysis Report"),
        dcc.Markdown(f'''
```
{report_text}
```
''')
    ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        html.H2("Visualizations"),
        html.Div([
            html.H3("Confusion Matrix"),
            html.Img(src=cm_image, style={'width': '100%'}) if cm_image else html.P("Confusion matrix not found.")
        ]),
        html.Div([
            html.H3("Raw Score Distribution"),
            html.Img(src=score_dist_image, style={'width': '100%'}) if score_dist_image else html.P("Score distribution not found.")
        ])
    ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'})
])

if __name__ == '__main__':
    app.run(debug=True)
