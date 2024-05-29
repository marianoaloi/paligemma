
let reconnectAttempts = 0;
const maxReconnectAttempts = 50; // Adjust as needed
const reconnectInterval = 5000; // 5 seconds between retries

const wsurl = "ws://192.168.25.90:8765/";

let pingTimeoutId;

var websocket = new WebSocket(wsurl);

function sendPing() {
    websocket.send(JSON.stringify({ action: 'ping' })); // Replace with your custom ping message

    pingTimeoutId = setTimeout(() => {
        console.error('Ping timeout! Assuming connection lost.');
        // Initiate reconnection logic (same as onclose handler)
        reconnect();
    }, 10000); // Timeout after 10 seconds
}

function reconnect() {
    if (reconnectAttempts < maxReconnectAttempts) {
        console.log('Attempting to reconnect...', reconnectAttempts + 1);
        reconnectAttempts++;

        setTimeout(() => {
            try {
                if (websocket.readyState != 1) {
                    websocket = new WebSocket(wsurl); // Re-create the WebSocket
                }
            } catch (error) {
                console.error('Reconnect error:', error);
                // Handle specific errors during reconnection (optional)
            }
        }, reconnectInterval);
    } else {
        console.error('Maximum reconnection attempts reached.');
        // Handle failed reconnection (e.g., display an error to the user)
    }
}



websocket.onopen = () => {
    console.log('WebSocket connection opened!');
    // (Optional) Set up initial ping interval (explained later)
    startPing();
};

const startPing = () => {
    websocket.send(JSON.stringify({ action: 'getAll' }));
    websocket.send(JSON.stringify({ action: 'resume' }));
    sendPing();
    setInterval(sendPing, 30000); // Ping every 30 seconds
}

websocket.onmessage = (event) => {
    console.log('Received message:', event.data);
    // Handle incoming messages as usual
};

websocket.onerror = (error) => {
    console.error('WebSocket error:', error);
    // (Optional) Handle specific errors (e.g., network errors)
};

websocket.onclose = (event) => {
    console.log('WebSocket connection closed:', event.code, event.reason);
    // Initiate reconnection logic
    reconnect();
};


window.addEventListener("DOMContentLoaded", () => {

    document.querySelector('#directory').addEventListener("change",
        async (event) => {
            try {
                await websocket.send(JSON.stringify({ action: 'readdir', directory: event.target.value }))
            } catch (Error) {
                console.error("websocket.readyState", websocket.readyState)
                console.error(Error)
            }
        }
    )

    var hiddeFilledStyle = ""
    document.querySelector('#hiddenFilled').addEventListener("click",
        async (event) => {
            if (!hiddeFilledStyle) {
                hiddeFilledStyle = "display:none"
            } else {
                hiddeFilledStyle = ""
            }
            Array.from(document.querySelectorAll('.description')).filter(txt => txt.value).map(txt => txt.parentElement.parentElement).forEach(txt => txt.style = hiddeFilledStyle)
        }
    )


    websocket.onmessage = ({ data }) => {
        const event = JSON.parse(data);
        switch (event.type) {
            case 'error':
                console.error(event.msg)
                break;
            case 'image':
                createImageBitmap(event.image,event.userId)
                break;
            case 'images':
                event.images.forEach(image => {
                    createImageBitmap(image,undefined)

                });
                break;
            case 'pong':
                clearTimeout(pingTimeoutId);
                break;
            case 'resume':
                resume(event);
                break;
            case 'id':
                console.log(`Id: ${event.id}`)
                id = event.id;
                break;
            default:
                break;
        }
    }
    table = document.querySelector('#images')



}
)

var id = undefined
var table = undefined

const createImageBitmap = (image,userId) => {
    if (text = document.querySelector(`#x${image._ItemList__id}`)) {
        if (this.id != userId)
            text.value = image.description

    } else {
        tr = document.createElement('tr')
        tr.innerHTML = `
    
    <td><img src="/img/?path=${image.pathPhoto}" class="photo" title="${image.pathPhoto}"></td>
    <td >
    <textarea onkeyup="editText(this)" 
    onchange="callResume()"
    path="${image.pathPhoto}" 
    id="x${image._ItemList__id}" 
    class="description"
    >${image.description}</textarea>
    </td>
    
    `
        table.appendChild(tr)
    }
    // console.log(image)
}

const editText = async (textAreaField) => {
    await websocket.send(JSON.stringify({
        action: 'write',
        image: textAreaField.getAttribute("path"),
        text: textAreaField.value
    }))
}

const callResume = async () => {
    await websocket.send(JSON.stringify({
        action: 'resume'
    }))
}

const resume = async (event) => {
    document.querySelector('#count').innerText = event.count
    document.querySelector('#total').innerText = event.total
    document.querySelector('#filled').innerText = event.filled
    document.querySelector('#empty').innerText = event.empty
}