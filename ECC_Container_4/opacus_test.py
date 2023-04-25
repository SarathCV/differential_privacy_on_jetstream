# #!/usr/bin/env python
# # encoding: utf-8
import json
# from flask import Flask
# app = Flask(__name__)
# @app.route('/')
# def index():
#     with open('/tmp/output1.txt') as f:
#         contents = f.read()
#     with open('/tmp/output2.txt') as f:
#         contents_1 = f.read()
#     with open('/tmp/output3.txt') as f:
#         contents_2 = f.read()
#     return json.dumps({"model_1":contents,"model_2":contents_1,"model_3":contents_2})
# app.run()
from flask import Flask
app = Flask(__name__)

@app.route('/')
# def hello_geek():
#     with open('/tmp/output1.txt') as f:
#         contents = f.read()
#     with open('/tmp/output2.txt') as f:
#         contents_1 = f.read()
#     with open('/tmp/output3.txt') as f:
#         contents_2 = f.read()
#     #return json.dumps({"model_1":contents,"model_2":contents_1,"model_3":contents_2})
#     combined_contents = {}
#     combined_contents.update(contents)
#     combined_contents.update(contents_1)
#     combined_contents.update(contents_2)

def print_data():
    with open('/tmp/output1.txt') as f:
        contents = f.read()
    contents = {'model_1':contents}
    with open('/tmp/output2.txt') as f:
        contents_1 = f.read()
    contents_1 = {'model_2':contents_1}
    with open('/tmp/output3.txt') as f:
        contents_2 = f.read()
    contents_2 = {'model_3':contents_2}
    #return json.dumps({"model_1":contents,"model_2":contents_1,"model_3":contents_2})
    data = {}
    data.update(contents)
    data.update(contents_1)
    data.update(contents_2)
    best_model = None
    best_accuracy = 0.0  # Initialize to a low value
    html = """
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #222222; /* dark background */
                color: #ffffff; /* text color */
                margin: 0;
                padding: 20px;
            }
            h1 {
                color: #ffffff;
                margin-top: 0;
                text-align: center;
            }
            .model-container {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-wrap: wrap;
                margin-top: 20px;
            }
            .model-card {
                background-color: #333333; /* card background color */
                border-radius: 5px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                margin: 10px;
                padding: 20px;
                max-width: 300px;
                width: 100%;
            }
            .model-card h2 {
                color: #ffffff;
                font-size: 24px;
                margin-top: 0;
            }
            .model-card p {
                color: #bbbbbb; /* text color */
                font-size: 18px;
                margin-bottom: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Results:</h1>
        <div class="model-container">
    """
    for model, values in data.items():
        loss, accuracy = map(float, values.strip().split("\n"))
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy
        html += """
            <div class="model-card">
                <h2>{}</h2>
                <p><strong>Accuracy</strong> {:.2f}</p>
                <p><strong>Loss:</strong> {:.2f}</p>
            </div>
        """.format(model, accuracy, loss)
    html += """
        </div>
        <h1>Best Model:</h1>
        <div class="model-container">
            <div class="model-card">
                <h2>{}</h2>
                <p><strong>Accuracy:</strong> {:.2f}</p>
                <p><strong>Loss:</strong> {:.2f}</p>
            </div>
        </div>
    </body>
    </html>
    """.format(best_model, float(data[best_model].strip().split("\n")[1]), float(data[best_model].strip().split("\n")[0]))
    return html

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)