# realtime_dog_sentiment_detection_using_cnn
# Real-time Dog Sentiment Detection Using CNN

A convolutional neural network (CNN) based project designed to detect and classify a dog's sentiment in real time from camera feeds or image inputs. This project aims to provide insights into a dog's emotional state by classifying its facial expressions or posture into categories such as "Happy," "Neutral," or "Anxious."

---

## Features

- **Real-time Detection**: Classifies dog sentiment from live video feeds.
- **Pre-trained Models**: Utilizes transfer learning on popular CNN architectures like ResNet or MobileNet.
- **Custom Dataset**: Trained on a dataset of labeled dog images with annotated sentiments.
- **User-friendly Interface**: Displays sentiment overlay on real-time video.

---

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or above
- pip (Python package installer)
- OpenCV (for real-time video processing)
- TensorFlow/Keras (for CNN model)
- NumPy and Matplotlib (for data handling and visualization)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/dog-sentiment-detection.git
   cd dog-sentiment-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download the pre-trained model weights:
   - Place the weights file (`model_weights.h5`) in the `models/` directory.

4. Run the application:
   ```bash
   python app.py
   ```

---

## Usage

### Real-time Sentiment Detection

1. Ensure your webcam is connected or a video feed is available.
2. Execute the following command:
   ```bash
   python app.py --mode live
   ```
3. Observe real-time sentiment predictions displayed on the video feed.

### Image Sentiment Classification

1. Place the image file in the `data/` directory.
2. Execute the following command:
   ```bash
   python app.py --mode image --input data/dog_image.jpg
   ```
3. View the sentiment result in the terminal or as an annotated output image saved in the `results/` directory.


## Acknowledgments

- [Kaggle Dog Emotion Dataset](https://www.kaggle.com/) for training data.
- TensorFlow/Keras for the deep learning framework.
- OpenCV for real-time video processing.
