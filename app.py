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

    topic = model.test(user_message)
    # bot_response = summarize_text(user_message)
    
    # return jsonify({'response': bot_response})
    return jsonify({'response': "Pz"+str(topic)})

@app.route('/train_model', methods=["POST"])
def train_model():
    data = request.get_json()

    
    train_data = data.get('train_data', [])
    num_topics = data.get('num_topics', 0)

    global model 
    model = PLSA(K=num_topics, maxIteration=30,threshold=10.0, topicWordsNum=5)
    p, Pz, lamda, theta, wordTop, id2w = model.train(dataset=train_data)

    for i in range(0,len(wordTop)):
        wordTop[i] = 'Pz'+str(i) + wordTop[i]

    

    result = dict(zip(wordTop, Pz))
    
    # Xử lý huấn luyện mô hình pLSA với dữ liệu này (tạm thời trả về dữ liệu nhận được)
    # result = {
    #     'train_data_len': len(train_data),
    #     'num_topics': 6,
    #     'status': 'Training successful (mock)'
    # }

    # Pz = [0.2, 0.4, 0.4]
    # result = {
    #     'Pz'+str(i)+'hihihohohaha': value for i, value in enumerate(Pz)
    # }

    # Trả kết quả về client
    return jsonify(result)

@app.route('/test_model', methods=['POST'])
def test_model():
    data = request.get_json()

    test_data = data.get('test_data', [])


    # test 
    lamda_new = [0.3, 0.4, 0.3]

    result = {'lamda'+str(i): value for i, value in enumerate(lamda_new)}

    return jsonify(result)
    # return lamda_new

def summarize_text(text):
    return text

if __name__ == '__main__':
    app.run(port=3000, debug=True)

