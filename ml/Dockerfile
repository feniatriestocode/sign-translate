# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:2.10.0-gpu

# Set working directory in container
WORKDIR /app

# Install necessary packages
RUN pip install --no-cache-dir \
    kaggle \
    kagglehub \
    mediapipe \
    opencv-python \
    numpy

# Copy the kaggle.json API key file to the container (you should already have this file)
COPY kaggle.json /root/.kaggle/kaggle.json

# Make sure the .kaggle folder has the correct permissions
RUN chmod 600 /root/.kaggle/kaggle.json

# Install additional requirements from the requirements.txt if you have it
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app

# Set the command to run your training script by default
CMD ["python", "train.py"]
