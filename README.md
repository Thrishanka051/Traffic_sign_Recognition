# Traffic_sign_Recognition
Here’s a **README.md** file content for your Traffic Sign Recognition project:

---

# **Traffic Sign Recognition System**

A real-time Traffic Sign Recognition system that integrates object detection and classification to detect traffic signs and classify them into nine predefined categories. This project aims to ensure safe navigation and adherence to traffic rules, making it ideal for autonomous vehicle systems.

---

## **Key Features**
- **Detection**: Identifies traffic signs in images or video streams using YOLO.
- **Classification**: Classifies detected signs into categories like speed limits, stop signs, and yield signs using a custom classification model.
- **Real-Time Processing**: Ensures quick decision-making for autonomous navigation.

---

## **Methodology**
The project consists of two main components: 
1. **Object Detection (YOLO)**: 
   - YOLO (You Only Look Once) was fine-tuned on a labeled dataset to detect traffic signs as a single class object.
   - The model was trained to identify the location of traffic signs in images or video streams efficiently.

2. **Classification Model**:
   - A custom classification model was developed from scratch using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.
   - The dataset, originally consisting of 43 classes, was reduced to 9 classes representing common traffic signs (e.g., stop, yield, speed limits).
   - This model classifies detected signs into one of the predefined categories.

3. **Real-Time Integration**:
   - The detected traffic sign regions are passed to the classification model for identification.
   - The system processes input from cameras in real time to provide actionable insights for autonomous systems.

---

## **Architecture Design**
The Traffic Sign Recognition System is designed with the following pipeline:

1. **Input**:  
   - Images or video frames captured by a car's camera.

2. **Traffic Sign Detection**:  
   - A YOLO model identifies traffic signs in the input and outputs bounding boxes around detected signs.

3. **Traffic Sign Classification**:  
   - The cropped bounding box regions are fed into the classification model, which assigns a label from the 9 predefined categories.

4. **Output**:  
   - The system generates the detected traffic sign class label (e.g., Speed Limit 50, Stop Sign) along with its position in the frame.

---

## **Dataset**
1. **Detection Dataset**:  
   - Custom labeled dataset for traffic sign detection. 
   - Labels include bounding boxes around traffic signs as a single object class.

2. **Classification Dataset**:  
   - **German Traffic Sign Recognition Benchmark (GTSRB)**.
   - Reduced to 9 classes for simplicity and task focus.

---

## **Setup Instructions**
1. Clone this repository:
   ```bash
   git clone https://github.com/Thrishanka051/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

---

## **Technologies Used**
- **YOLO**: Object detection for traffic signs.
- **TensorFlow/Keras**: For building and training the custom classification model.
- **OpenCV**: For real-time video processing.
- **Python**: For data preprocessing, training, and integration.

---



## **Acknowledgments**
- The **GTSRB dataset** for traffic sign classification.
- The YOLO framework for efficient object detection.

--- 
