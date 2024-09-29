function sendMessage() {
    const input = document.getElementById('messageInput');
    const chatBox = document.getElementById('chatBox');
    const chatFile = document.getElementById('chatFile').files[0]; 
    const messageText = input.value.trim();

    if (messageText) {
        // Thêm tin nhắn của người dùng vào chat box
        const userMessage = document.createElement('div');
        userMessage.className = 'message user';
        userMessage.innerHTML = `${messageText}<span class="timestamp">${formatTime(new Date())}</span>`;
        chatBox.appendChild(userMessage);
        
        // Gửi tin nhắn đến server bằng fetch
        fetch('/send_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ test_data: messageText }),
        })
        .then(response => response.json())
        .then(data => {
            // Thêm tin nhắn của bot vào chat box
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot';
            botMessage.innerHTML = `${data.response}<span class="timestamp">${formatTime(new Date())}</span>`;
            chatBox.appendChild(botMessage);

            // Cuộn xuống tin nhắn mới nhất
            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
        });

        // Làm sạch ô nhập liệu
        input.value = '';
    }
    else if (chatFile) {
        
        readFileAsList(chatFile, function(fileData) {
            // Thêm tin nhắn của người dùng vào chat box
        const userMessage = document.createElement('div');
        userMessage.className = 'message user';
        userMessage.innerHTML = `${fileData}<span class="timestamp">${formatTime(new Date())}</span>`;
        chatBox.appendChild(userMessage);
            // Gửi dữ liệu test (file) lên server
            fetch('/send_message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ test_data: fileData }),
            })
            .then(response => response.json())
            .then(data => {
                // Thêm tin nhắn của bot vào chat box
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot';
            botMessage.innerHTML = `${data.response}<span class="timestamp">${formatTime(new Date())}</span>`;
            chatBox.appendChild(botMessage);

            // Cuộn xuống tin nhắn mới nhất
            chatBox.scrollTop = chatBox.scrollHeight;
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });

    } else {
        alert('Vui lòng chọn file hoặc nhập nội dung text.');
    }
}

function formatTime(date) {
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
}

// Thêm sự kiện lắng nghe cho phím Enter
document.getElementById('messageInput').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Ngăn chặn hành vi mặc định của Enter (như xuống dòng)
        sendMessage(); // Gửi tin nhắn
    }
});

// Đọc file txt và chuyển đổi nội dung thành list
function readFileAsList(file, callback) {
    const reader = new FileReader();
    reader.onload = function(event) {
        const text = event.target.result;
        // Chuyển đổi văn bản thành một danh sách bằng cách tách theo dòng
        const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
        callback(lines); // Gọi lại với dữ liệu dạng list
    };
    reader.readAsText(file);
}

// // Add event listener for file input
// document.getElementById('chatFile').addEventListener('change', function() {
// const fileName = this.files[0] ? this.files[0].name : '';
// const messageInput = document.getElementById('messageInput');
// messageInput.value = fileName;
// });


// Gửi file test hoặc text nhập lên Flask
function sendTestData() {
    const chatFile = document.getElementById('chatFile').files[0];   // File hoặc nội dung bên phải
    const messageInput = document.getElementById('messageInput').value.trim(); // Nội dung bên phải

        // Hàm thêm tin nhắn vào chat box
    const addMessageToChat = (messageText, sender) => {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${sender}`;
        messageElement.innerHTML = `${messageText}<span class="timestamp">${formatTime(new Date())}</span>`;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight; // Cuộn xuống tin nhắn mới nhất
    };

    // Nếu người dùng đã chọn file
    if (chatFile) {
        readFileAsList(chatFile, function(fileData) {
            // Gửi dữ liệu test (file) lên server
            fetch('/test_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ test_data: fileData }),
            })
            .then(response => response.json())
            .then(data => {
                addMessageToChat(data.response, 'user')

                alert(data.message);
                console.log(data);  // Xem phản hồi từ server
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });

    } 
    // Nếu người dùng chỉ nhập văn bản
    else if (messageInput) {
        const textData = messageInput.split(/\r?\n/).filter(line => line.trim() !== ''); // Tách nội dung text thành list

        // Gửi dữ liệu test (text) lên server
        fetch('/test_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ test_data: textData }),
        })
        .then(response => response.json())
        .then(data => {
            addMessageToChat(data.message, 'user')

            alert(data.message);
            console.log(data);  // Xem phản hồi từ server
        })
        .then(result =>{
            // for(let key in result){
            //     alert(result.key)
            // }
            alert(result.lamda0)
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        alert('Vui lòng chọn file hoặc nhập nội dung text.');
    }

    sendMessage();
}


function sendTrainData() {
    const trainFileInput = document.getElementById('trainFile');
    const numTopicsInput = document.getElementById('numTopics');
    
    if (trainFileInput.files.length > 0 && numTopicsInput.value) {
        const trainFile = trainFileInput.files[0];
        const numTopics = parseInt(numTopicsInput.value, 10);
        
        // Read the train file and process it
        const reader = new FileReader();
        reader.onload  = function(event) {
            const trainData = event.target.result.split('\n').map(line => line.trim());; // Assuming each line is a separate training example
            
            // Prepare the data to send
            const dataToSend = {
                train_data: trainData,
                num_topics: numTopics
            };

            // Send the data to the server
            fetch('/train_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(dataToSend)
            })
            .then(response => response.json())
            .then(result => {

                // Tạo labels và chú thích
                const labels = [];
                const tooltips = [];
                const data = [];

                for (const [key, value] of Object.entries(result)) {
                    const label = key.slice(0, 3); // Lấy 3 ký tự đầu làm tên cột
                    const tooltip = key.slice(3);  // Phần còn lại làm chú thích
                    labels.push(label);
                    tooltips.push(tooltip);
                    data.push(value);
                }

                // Vẽ biểu đồ cột
                drawChart(labels, data, tooltips);

                // Hiển thị bảng chú thích dưới đồ thị
                createLegend(labels, tooltips);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        };

        // Read the file as text
        reader.readAsText(trainFile);
    } else {
        alert('Vui lòng chọn file train và nhập số chủ đề.');
    }
}

function drawChart(labels, data, tooltips) {
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Topic Probability Distribution (Pz)',
                data: data,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(tooltipItem) {
                            // Sử dụng tooltip để hiển thị chú thích
                            return `${tooltips[tooltipItem.dataIndex]}: ${tooltipItem.formattedValue}`;
                        }
                    }
                }
            }
        }
    })};

    function createLegend(labels, tooltips) {
        const legendContainer = document.getElementById('legend');
        
        // Xóa nội dung cũ trong bảng chú thích nếu có
        legendContainer.innerHTML = '';
        
        // Tạo bảng chú thích
        const table = document.createElement('table');
        table.style.margin = '20px auto';
        table.style.borderCollapse = 'collapse';
        table.style.textAlign = 'center';
        table.style.width = '50%';
        table.style.fontSize = '0.8em'; // Giảm kích thước font chữ
        // Tạo hàng tiêu đề
        const headerRow = document.createElement('tr');
        const headerLabel = document.createElement('th');
        headerLabel.textContent = 'Topics';
        const headerTooltip = document.createElement('th');
        headerTooltip.textContent = 'Word Top';
        

        headerRow.appendChild(headerLabel);
        headerRow.appendChild(headerTooltip);
        table.appendChild(headerRow);
    
        // Thêm các dòng dữ liệu
        labels.forEach((label, index) => {
            const row = document.createElement('tr');
            row.style.borderBottom = '1px solid #ccc';
    
            const labelCell = document.createElement('td');
            labelCell.textContent = label;
            labelCell.style.padding = '10px';
            
            // labelCell.style.padding = '20px';
            // tooltipCell.style.padding = '35px';

            const tooltipCell = document.createElement('td');
            tooltipCell.textContent = tooltips[index];
            tooltipCell.style.padding = '10px';
            
            row.appendChild(labelCell);
            row.appendChild(tooltipCell);
            table.appendChild(row);
        });
    
        // Thêm bảng vào container chú thích
        legendContainer.appendChild(table);
    }


// Function to save the trained model
function saveModel() {
    fetch('/save_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        alert('Model đã được lưu thành công: ' + data.model_name);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Function to upload a model file
function uploadModel() {
    const modelFileInput = document.getElementById('modelFile');
    
    if (modelFileInput.files.length > 0) {
        const modelFile = modelFileInput.files[0];
        
        const formData = new FormData();
        formData.append('model_file', modelFile);

        fetch('/upload_model', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(result => {

            // Tạo labels và chú thích
            const labels = [];
            const tooltips = [];
            const data = [];

            for (const [key, value] of Object.entries(result)) {
                const label = key.slice(0, 3); // Lấy 3 ký tự đầu làm tên cột
                const tooltip = key.slice(3);  // Phần còn lại làm chú thích
                labels.push(label);
                tooltips.push(tooltip);
                data.push(value);
            }

            // Vẽ biểu đồ cột
            drawChart(labels, data, tooltips);

            // Hiển thị bảng chú thích dưới đồ thị
            createLegend(labels, tooltips);
        })
        .then(data => {
            alert('Model đã được tải lên và sẵn sàng sử dụng.');
        })
        
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        alert('Vui lòng chọn file model để tải lên.');
    }
}

// Function to enable the test inputs (messageInput and button)
function enableTestInputs() {
    document.getElementById('messageInput').disabled = false;
    document.querySelector('button[onclick="sendMessage()"]').disabled = false;
    document.getElementById('chatFile').disabled = false;
}   