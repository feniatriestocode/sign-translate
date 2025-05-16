// document.addEventListener('DOMContentLoaded', () => {
//     // DOM elements
//     const webcamElement = document.getElementById('webcam');
//     const canvasElement = document.getElementById('canvas');
//     const canvasCtx = canvasElement.getContext('2d');
//     const predictionElement = document.getElementById('prediction');
//     const startButton = document.getElementById('startButton');
//     const loadingElement = document.getElementById('loading');
//     const errorElement = document.getElementById('error');
//     const statusElement = document.getElementById('status');

//     const drawingUtils = window;
    
//     // State variables
//     let model = null;
//     let hands = null;
//     let isRunning = false;
//     let stream = null;
//     let scaler = null;

// async function loadScaler() {
//     try {
//         const response = await fetch('minmax_scaler.json');
//         const scalerParams = await response.json();
        
//         // Create a scaler function that matches sklearn's MinMaxScaler
//         scaler = {
//             min_: tf.tensor1d(scalerParams.min_),
//             scale_: tf.tensor1d(scalerParams.scale_),
//             transform: function(data) {
//                 const dataTensor = tf.tensor2d(data);
//                 // X_std = (X - X.min) / (X.max - X.min)
//                 // X_scaled = X_std * (max - min) + min
//                 const scaled = dataTensor.sub(this.min_).mul(this.scale_);
//                 dataTensor.dispose();
//                 return scaled.arraySync();
//             }
//         };
//         console.log("Scaler loaded successfully");
//     } catch (error) {
//         console.error("Error loading scaler:", error);
//         // Fallback to no transformation if scaler fails
//         scaler = {
//             transform: (data) => data
//         };
//     }
// }


//     // Check WebGL support
//     function checkWebGLSupport() {
//         try {
//             const canvas = document.createElement('canvas');
//             const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
//             return !!gl;
//         } catch (e) {
//             return false;
//         }
//     }

//     // Initialize the application
//     async function initialize() {
//         if (!checkWebGLSupport()) {
//             showError("WebGL is not supported in your browser. Please try Chrome or Firefox with hardware acceleration enabled.");
//             startButton.disabled = true;
//             return;
//         }

//         // Set TensorFlow.js backend
//         await tf.setBackend('webgl');
//         statusElement.textContent = "Using WebGL backend";

//         // Load models
//         await loadModels();
//     }

//     // Load TensorFlow.js and MediaPipe models
//     async function loadModels() {
//         try {
//             loadingElement.style.display = 'block';
//             startButton.disabled = true;
            
//             // Load TensorFlow model
//             model = await tf.loadLayersModel('model.json');
//             console.log("TensorFlow model loaded");
            
//             // Load scaler
//             await loadScaler();
//             console.log("Scaler initialized");
            
//             // Initialize MediaPipe Hands
//             hands = new window.Hands({
//                 locateFile: (file) => {
//                     return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
//                 }
//             });
            
//             hands.setOptions({
//                 maxNumHands: 1,
//                 modelComplexity: 1,
//                 minDetectionConfidence: 0.5,
//                 minTrackingConfidence: 0.5
//             });
            
//             hands.onResults(onHandsResults);
            
//             console.log("MediaPipe Hands initialized");
//             loadingElement.style.display = 'none';
//             startButton.disabled = false;
//             statusElement.textContent = "Models and scaler loaded successfully";
//         } catch (error) {
//             console.error("Error loading models or scaler:", error);
//             showError(`Failed to load models or scaler: ${error.message}`);
//             loadingElement.style.display = 'none';
//         }
//     }


    
//     // Process results from MediaPipe Hands
//     function onHandsResults(results) {
//         if (!isRunning) return;

//         // Draw hand landmarks
//         canvasCtx.save();
//         canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//         if (results.multiHandLandmarks) {
//             for (const landmarks of results.multiHandLandmarks) {
//                 window.drawConnectors(
//                     canvasCtx, landmarks, window.HAND_CONNECTIONS,
//                     {color: '#00FF00', lineWidth: 2});
//                 window.drawLandmarks(
//                     canvasCtx, landmarks,
//                     {color: '#FF0000', lineWidth: 1});
//             }
            
//             // Extract landmarks and make prediction
//             const landmarks = extractLandmarks(results);
//             if (landmarks) {
//                 makePrediction(landmarks);
//             }
//         }
//         canvasCtx.restore();
//     }

//     // Extract landmarks in the format expected by the model
//     function extractLandmarks(results) {
//         if (results.multiHandLandmarks?.length > 0) {
//             const hand = results.multiHandLandmarks[0];
//             return hand.flatMap(landmark => [landmark.x, landmark.y, landmark.z]);
//         }
//         return null;
//     }

//     function preprocessLandmarks(landmarks) {
//         // 1. Check for invalid/missing landmarks
//         if (!landmarks || landmarks.length !== 63) { // 21 landmarks * 3 coords
//             console.warn("Invalid landmarks detected");
//             return null;
//         }
        
//         // 2. Optional: Simple low-pass filter to reduce jitter
//         if (!window.landmarkHistory) window.landmarkHistory = [];
//         window.landmarkHistory.push(landmarks);
//         if (window.landmarkHistory.length > 5) window.landmarkHistory.shift();
        
//         // 3. Average the last few frames
//         const smoothed = window.landmarkHistory[0].map((_, i) => {
//             return window.landmarkHistory.reduce((sum, lm) => sum + lm[i], 0) / window.landmarkHistory.length;
//         });
        
//         return smoothed;
//     }
    

//     // Make prediction using the TensorFlow model
//     // async function makePrediction(landmarks) {
//     //     try {
//     //         // Convert to tensor and make prediction
//     //         const inputTensor = tf.tensor2d([landmarks]);
//     //         const prediction = model.predict(inputTensor);
//     //         const values = await prediction.data();
//     //         inputTensor.dispose();
//     //         prediction.dispose();
            
//     //         // Get the predicted class
//     //         const predictedClass = values.indexOf(Math.max(...values));
            
//     //         // Update UI with prediction
//     //         updatePrediction(predictedClass);
//     //     } catch (error) {
//     //         console.error("Prediction error:", error);
//     //     }
//     // }

//     async function makePrediction(landmarks) {
//         try {
//             // 1. Ensure landmarks are in correct order (same as training)
//             const orderedLandmarks = [];
//             for (let i = 0; i < 21; i++) { // 21 landmarks
//                 orderedLandmarks.push(
//                     landmarks[i*3],     // x
//                     landmarks[i*3+1],   // y
//                     landmarks[i*3+2]    // z
//                 );
//             }
            
//             // 2. Apply the same scaling used during training
//             const scaledLandmarks = scaler.transform([orderedLandmarks])[0];
            
//             // 3. Convert to tensor and make prediction
//             const inputTensor = tf.tensor2d([scaledLandmarks]);
//             const prediction = model.predict(inputTensor);
//             const values = await prediction.data();
            
//             // Clean up
//             inputTensor.dispose();
//             prediction.dispose();
            
//             // Get predicted class
//             const predictedClass = values.indexOf(Math.max(...values));
//             updatePrediction(predictedClass);
//         } catch (error) {
//             console.error("Prediction error:", error);
//         }
//     }

//     // Update the prediction display
//     function updatePrediction(classIndex) {
//         // Replace this with your actual class labels
//         const classLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
//                            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
//                            'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space'];
        
//         if (classIndex >= 0 && classIndex < classLabels.length) {
//             predictionElement.textContent = classLabels[classIndex];
//         } else {
//             predictionElement.textContent = "Unknown";
//         }
//     }

//     // Start the detection process
//     async function startDetection() {
//         if (isRunning) return;
        
//         try {
//             isRunning = true;
//             startButton.textContent = "Stop Detection";
//             webcamElement.style.display = 'block';
            
//             // Get webcam stream
//             stream = await navigator.mediaDevices.getUserMedia({
//                 video: {
//                     facingMode: 'user',
//                     width: { ideal: 640 },
//                     height: { ideal: 480 }
//                 },
//                 audio: false
//             });
            
//             webcamElement.srcObject = stream;
            
//             // Start processing frames
//             processFrame();
//         } catch (error) {
//             console.error("Error starting detection:", error);
//             showError(`Could not access camera: ${error.message}`);
//             stopDetection();
//         }
//     }

//     // Stop the detection process
//     function stopDetection() {
//         isRunning = false;
//         startButton.textContent = "Start Detection";
//         webcamElement.style.display = 'none';
//         canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//         predictionElement.textContent = "Ready";
        
//         if (stream) {
//             stream.getTracks().forEach(track => track.stop());
//             stream = null;
//         }
//     }

//     // Process each video frame
//     async function processFrame() {
//         if (!isRunning) return;
        
//         try {
//             if (webcamElement.readyState === webcamElement.HAVE_ENOUGH_DATA) {
//                 // Resize canvas to match video dimensions
//                 canvasElement.width = webcamElement.videoWidth;
//                 canvasElement.height = webcamElement.videoHeight;
                
//                 // Send frame to MediaPipe Hands
//                 await hands.send({image: webcamElement});
//             }
            
//             // Continue processing frames
//             requestAnimationFrame(processFrame);
//         } catch (error) {
//             console.error("Frame processing error:", error);
//             stopDetection();
//         }
//     }

//     // Toggle detection on button click
//     startButton.addEventListener('click', () => {
//         if (isRunning) {
//             stopDetection();
//         } else {
//             startDetection();
//         }
//     });

//     // Show error message
//     function showError(message) {
//         errorElement.textContent = message;
//         errorElement.style.display = 'block';
//     }

//     // Initialize the app
//     initialize();
// });

document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');
    const canvasCtx = canvasElement.getContext('2d');
    const predictionElement = document.getElementById('prediction');
    const startButton = document.getElementById('startButton');
    const loadingElement = document.getElementById('loading');
    const errorElement = document.getElementById('error');
    const statusElement = document.getElementById('status');

    const drawingUtils = window;
    
    // State variables
    let model = null;
    let hands = null;
    let isRunning = false;
    let stream = null;

    // Check WebGL support
    function checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            return !!gl;
        } catch (e) {
            return false;
        }
    }

    // Initialize the application
    async function initialize() {
        if (!checkWebGLSupport()) {
            showError("WebGL is not supported in your browser. Please try Chrome or Firefox with hardware acceleration enabled.");
            startButton.disabled = true;
            return;
        }

        // Set TensorFlow.js backend
        await tf.setBackend('webgl');
        statusElement.textContent = "Using WebGL backend";

        // Load models
        await loadModels();
    }

    // Load TensorFlow.js and MediaPipe models
    async function loadModels() {
        try {
            loadingElement.style.display = 'block';
            startButton.disabled = true;
            
            // Load TensorFlow model
            model = await tf.loadLayersModel('model.json');
            console.log("TensorFlow model loaded");
            
            // Initialize MediaPipe Hands
            hands = new window.Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            });
            
            hands.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            hands.onResults(onHandsResults);
            
            console.log("MediaPipe Hands initialized");
            loadingElement.style.display = 'none';
            startButton.disabled = false;
            statusElement.textContent = "Models loaded successfully";
        } catch (error) {
            console.error("Error loading models:", error);
            showError(`Failed to load models: ${error.message}`);
            loadingElement.style.display = 'none';
        }
    }

    // Process results from MediaPipe Hands
    function onHandsResults(results) {
        if (!isRunning) return;

        // Draw hand landmarks
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        if (results.multiHandLandmarks) {
            for (const landmarks of results.multiHandLandmarks) {
                window.drawConnectors(
                    canvasCtx, landmarks, window.HAND_CONNECTIONS,
                    {color: '#00FF00', lineWidth: 2});
                window.drawLandmarks(
                    canvasCtx, landmarks,
                    {color: '#FF0000', lineWidth: 1});
            }
            
            // Extract landmarks and make prediction
            const landmarks = extractLandmarks(results);
            if (landmarks) {
                makePrediction(landmarks);
            }
        }
        canvasCtx.restore();
    }

    // Extract landmarks in the format expected by the model
    function extractLandmarks(results) {
        if (results.multiHandLandmarks?.length > 0) {
            const hand = results.multiHandLandmarks[0];
            return hand.flatMap(landmark => [landmark.x, landmark.y, landmark.z]);
        }
        return null;
    }

    // Make prediction using the TensorFlow model
    async function makePrediction(landmarks) {
        try {
            // Convert to tensor and make prediction
            const inputTensor = tf.tensor2d([landmarks]);
            const prediction = model.predict(inputTensor);
            const values = await prediction.data();
            inputTensor.dispose();
            prediction.dispose();
            
            // Get the predicted class
            const predictedClass = values.indexOf(Math.max(...values));
            
            // Update UI with prediction
            updatePrediction(predictedClass);
        } catch (error) {
            console.error("Prediction error:", error);
        }
    }

    // Update the prediction display
    function updatePrediction(classIndex) {
        // Replace this with your actual class labels
        const classLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                           'U', 'V', 'W', 'X', 'Y', 'Z', 'Nothing', 'Space'];
        
        if (classIndex >= 0 && classIndex < classLabels.length) {
            predictionElement.textContent = classLabels[classIndex];
        } else {
            predictionElement.textContent = "Unknown";
        }
    }

    // Start the detection process
    async function startDetection() {
        if (isRunning) return;
        
        try {
            isRunning = true;
            startButton.textContent = "Stop Detection";
            webcamElement.style.display = 'block';
            
            // Get webcam stream
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });
            
            webcamElement.srcObject = stream;
            
            // Start processing frames
            processFrame();
        } catch (error) {
            console.error("Error starting detection:", error);
            showError(`Could not access camera: ${error.message}`);
            stopDetection();
        }
    }

    // Stop the detection process
    function stopDetection() {
        isRunning = false;
        startButton.textContent = "Start Detection";
        webcamElement.style.display = 'none';
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        predictionElement.textContent = "Ready";
        
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    // Process each video frame
    async function processFrame() {
        if (!isRunning) return;
        
        try {
            if (webcamElement.readyState === webcamElement.HAVE_ENOUGH_DATA) {
                // Resize canvas to match video dimensions
                canvasElement.width = webcamElement.videoWidth;
                canvasElement.height = webcamElement.videoHeight;
                
                // Send frame to MediaPipe Hands
                await hands.send({image: webcamElement});
            }
            
            // Continue processing frames
            requestAnimationFrame(processFrame);
        } catch (error) {
            console.error("Frame processing error:", error);
            stopDetection();
        }
    }

    // Toggle detection on button click
    startButton.addEventListener('click', () => {
        if (isRunning) {
            stopDetection();
        } else {
            startDetection();
        }
    });

    // Show error message
    function showError(message) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }

    // Initialize the app
    initialize();
});