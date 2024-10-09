import os
from model import PLSA
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = None

@app.route('/', methods=["GET"])
def hello_world():
    return render_template("index.html")

@app.route('/send_message', methods=["POST"])
def handle_message():
    data = request.get_json()
    user_message = data.get('test_data')

    global model

    if model is None:
        return jsonify({'response': "Vui lòng train model hoặc up model PLSA!!!"})

    # If user_message is a list, you might want to concatenate or process it differently
    if isinstance(user_message, list):
        # Process the list as needed
        topics = model.test(user_message)  # Modify accordingly if your model can handle lists
        return jsonify({'response': str(topics)})
    else:
        # Handle the case where user_message is a single string
        topic = model.test([user_message])  # Treat it as a single item list
        return jsonify({'response': str(topic)})


@app.route('/train_model', methods=["POST"])
def train_model():
    data = request.get_json()

    
    train_data = data.get('train_data', [])
    num_topics = data.get('num_topics', 0)

    global model 
    model = PLSA(K=num_topics, maxIteration=30,threshold=10.0, topicWordsNum=7)
    p, Pz, lamda, theta, wordTop, id2w = model.train(dataset=train_data)

    for i in range(0,len(wordTop)):
        wordTop[i] = 'Pz'+str(i) + wordTop[i]

    

    result = dict(zip(wordTop, Pz))
    return jsonify(result)


def summarize_text(text):
    return text

# Save model endpoint
@app.route('/save_model', methods=['POST'])
def save_model():
    model_name = 'saved_plsa_model.pkl'
    
    global model
    model.save_model(model_name)


    return jsonify({'model_name': model_name})

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files:
        return jsonify({'error': 'No model file uploaded'}), 400

    model_file = request.files['model_file']
    

    root_dir = app.root_path
    file_path = os.path.join(root_dir, model_file.filename)
    # Save the uploaded model to the server
    model_file.save(file_path)
    global model
    
    try:
        # Load the model using the PLSA class instance
        model = PLSA(K=3, maxIteration=30,threshold=10.0, topicWordsNum=5)
        p, Pz, lamda, theta, wordTop, id2w = model.load_model(file_path)
        for i in range(0,len(wordTop)):
            wordTop[i] = 'Pz'+str(i) + wordTop[i]

    

        result = dict(zip(wordTop, Pz))
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

    return jsonify(result)



if __name__ == '__main__':
    app.run(port=3000, debug=True)

