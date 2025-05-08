// Select the video element
const webcamElement = document.getElementById('webcam');

// Load the TensorFlow.js model
let model;
async function loadModel() {
    model = await tf.loadLayersModel('model.json'); // path to your model
    console.log('Model loaded');
}

// Initialize Mediapipe Hands
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

// Setup the hands model and process the webcam video frames
async function setupWebcam() {
    return new Promise((resolve, reject) => {
        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
            webcamElement.srcObject = stream;
            webcamElement.addEventListener('loadeddata', resolve, false);
        }).catch(reject);
    });
}

// Process webcam frames for hand landmarks
async function processFrame() {
    const image = await tf.browser.fromPixels(webcamElement);
    const results = await hands.send({ image });

    if (results.multiHandLandmarks) {
        // Process each hand's landmarks
        for (const landmarks of results.multiHandLandmarks) {
            // Convert landmarks to the format expected by the model
            const inputData = [];
            for (const landmark of landmarks) {
                inputData.push(landmark.x, landmark.y, landmark.z); // 3D coordinates
            }

            // Prepare input for the model (TensorFlow.js requires a tensor)
            const inputTensor = tf.tensor([inputData]);

            // Run inference with the model
            const prediction = await model.predict(inputTensor);
            const predictedClass = prediction.argMax(-1).dataSync()[0];
            console.log('Predicted Class: ', predictedClass);
        }
    }
    requestAnimationFrame(processFrame); // Keep processing the next frame
}

// Run everything
async function run() {
    await loadModel();
    await setupWebcam();
    processFrame(); // Start processing webcam frames
}

run();
