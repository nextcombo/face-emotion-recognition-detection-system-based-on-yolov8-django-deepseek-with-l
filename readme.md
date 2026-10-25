# Face Emotion Recognition Detection System Based on Yolov8+Django+DeepSeek with Login Interface in Python Source Code + ONNX Model + Beautiful Web Interface

Effect Display: Login interface main interface based on YOLOv8 target detection system, supports image, video and real-time camera detection, integrated DeepSeek.

## Effect Display

Login interface

![](01-73b4c016.jpg)

Main Interface

![](02-14282c38.jpg)

A YOLOv8-based object detection system, supporting image, video, and real-time camera detection, integrated with DeepSeek AI analysis functions.

### Functional Characteristics

-

Image target detection

-

Video file detection

-

Real-time camera detection

-

DeepSeek AI Intelligent Analysis

-

Adjustable detection parameters (confidence, IoU)

-

Beautiful user interface

### [Test Environment]

windows10

anaconda3+python3.10

torch == 2.3.1

ultralytics == 8.4.7

Django == 5.2.11

onnxruntime == 1.17.0

### Model can detect 8 categories

names: ['anger', 'content', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

### Training dataset introduction

Dataset format: YOLO format (txt file without split path, only contains jpg images and corresponding yolo format txt files)

Number of images (number of jpg files): 9400

Number of annotations (number of txt files): 9400

Number of training set: 6586

Number of validation set: 1873

Test set size: 941

Annotation category count: 8

Label category names (note that the order of categories in the YOLO format does not correspond to this, but is based on the 'labels' folder with 'classes.txt'): ['anger', 'content', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

Number of boxes labeled in each category:

anger (anger) frame count = 1191

Content (satisfaction) box count = 1149

Disgust (disgust) frame count = 1172

Fear Frame Count = 1184

Happy (happy) frame count = 1173

Neutral (neutral) frame count = 1232

sad (sadness) frame count = 1192

Surprise (surprise) box count = 1245

Total frame count = 9538

Using labelImg:

Annotation rules: Draw a rectangle around the category.

Important Notice: No

Special : This dataset does not guarantee the accuracy of the trained models or weight files.

![](03-0fca15b9.jpg)

Annotation example:

![](04-e4d51812.jpg)

### Training dataset information

Training set image count = 6586, Validation set image count = 1873, Training map = 76.6%, Training precision (Precision) = 64.1%, Training recall (Recall) = 75.4%

### Install Dependencies

```
pip install -r requirements.txt
```

### Database Migration

```
python manage.py makemigrations
python manage.py migrate
```

### Running Projects

```
python manage.py runserver
```

Visit http://127.0.0.1:8000

### Default login information

-

Username: admin

-

Password: admin

### Main File Structure of the Project

```
yolov8-django-deepseek-det-system/
├── manage.py                 # Django Project Management Script
├── requirements.txt  # Project dependency package list
├── README.md  # Project Description Document
├── index.html  # Main page (as a backup or example page)
├── start_lan.bat  # A one-click batch script for starting the system
├── yolo_detection/ # Django project configuration directory
├── __init__.py
├── settings.py # Project configuration file (database, API key, etc.)
├── urls.py # Project's total route configuration
└── wsgi.py # WSGI deployment configuration
├── detection/ Detection function application module
├── __init__.py
├── apps.py # Application configuration
├── forms.py # Form definition (such as file upload form)
├── models.py
├── urls.py # Detection of application route configuration
└── views.py # Core view functions (YOLO detection logic, file processing, etc.)
├── templates/ # Web page template directory
├── index.html # Main operation interface (upload, detection, result display)
└── login.html # Login Page
├── media/ # Media file storage directory
└── uploads/ # Directory for storing uploaded files by users
├── test_img/  # Test image directory
└── *.jpg # Detection sample images
├── weights/  # Model directory
```

### Technology Stack

-

Django 4.2+

-

YOLOv8 (Ultralytics)

-

OpenCV

-

DeepSeek API

-

Chart.js

-

HTML5/CSS3/JavaScript

### Notes

-

Ensure that the `yolov8n.onnx` model file is in the project's weights directory.

-

DeepSeek API key needs to be configured in `settings.py`

-

Recommend using Python 3.8+

### Common Evaluation Parameters Introduction

In the task of object detection, evaluating the performance of a model is crucial. The terms you mentioned are commonly used metrics for assessing the performance of a model. Here are detailed explanations of these terms:

Class: This usually refers to the category of objects that the model is designed to detect. For example, a model may be trained to detect different categories of objects such as vehicles, pedestrians, or animals.

Images: Represents the number of images in the validation set. The validation set is a dataset used to evaluate model performance, separate from the training set to ensure fairness in evaluation results.

Instances: The total number of target objects in all images. This includes the sum of all category objects, for example, if the validation set contains 100 images, each image has an average of 5 target objects, then Instances is 500.

P (Precision): Precision is the ratio of true positives among the instances predicted as positive by the model. The formula is: Precision = TP / (TP + FP), where TP represents True Positives and FP represents False Positives.

R (Recall): Recall is the proportion of all true positives correctly predicted by the model. The formula is: Recall = TP / (TP + FN), where FN represents false negatives.

mAP50: Mean Average Precision at IoU (Intersection over Union) threshold of 0.5, which represents the average precision when the IoU (Intersection over Union) is set to 0.5. The IoU metric measures the degree of overlap between predicted bounding boxes and ground truth bounding boxes. mAP is a comprehensive metric that considers both precision and recall, used to evaluate the performance of models across different recall levels. At an IoU of 0.5, if the overlap between the predicted bounding box and the ground truth bounding box reaches or exceeds 50%, it is considered correct.

mAP50-95: Refers to the average precision of a model within the range of IoU from 0.5 to 0.95 (with a step of 0.05). This is a more stringent evaluation criterion, demanding higher overlap between predicted bounding boxes and true ones. In object detection tasks, a higher IoU threshold implies that the model needs to be more accurate in target object localization. The computation of mAP50-95 takes into account multiple IoU thresholds ranging from lenient to strict, thus providing a comprehensive assessment of the model's performance. These metrics collectively form an important framework for evaluating the performance of object detection models. By comparing the performance of different models on these metrics, one can determine which model may be more effective in practical applications.

### Common Issues

In the training of target detection, if the Mean Average Precision (MAP) is low, there may be the following reasons:

Reason 1: Underfitting: If the amount of training data is too small, the model may not learn enough features to affect prediction results, leading to underfitting and consequently causing a low MAP. Therefore, increasing the size of the dataset can be beneficial.

Reason two: Small targets: If the dataset contains a large proportion of small targets, there is a possibility that the map will be low because the features of small targets are not obvious, making it difficult for the model to learn them.

Reason 3: Mismatched model tuning parameters: For instance, if the learning rate is adjusted too high, it may lead to an overly rapid learning capability, resulting in disordered adjustments of model parameters.

Reason 4: Overfitting (This is now a rare occurrence in models): If the model performs exceptionally well on training data but poorly on validation or test data, it may be due to overfitting. This is often caused by an excessive number of model parameters relative to the amount of training data available, leading the model to learn noise or specific patterns from the training data and fail to generalize to new data. Nowadays, target detection models are well-equipped to handle this situation, with few instances of overfitting occurring.

Reason 5: Different scenes: The validation set has high accuracy, but the test set does not, which may be due to inconsistencies between the training model's scene images and those in the test map. To address these issues, the following measures can be taken to improve MAP:

(1) Optimize model structure: Select the appropriate model based on the characteristics of the task and dataset, and try using different network architectures and components to improve model performance.

(2) Enhance Data Preprocessing: Proper preprocessing and enhancement of data, such as data normalization, missing value imputation, data expansion, etc., to enhance the generalization capability of the model.

(3) Adjusting the Loss Function: Try using different loss functions or combining multiple loss functions to optimize model performance.

(4) Optimize training strategies: Adjust hyperparameters such as learning rate, batch size, and number of training iterations, and employ optimization algorithms like learning rate decay and momentum to enhance model training effectiveness.

(5) Use pre-trained models: Transfer learning using pre-trained models on large datasets can accelerate model convergence and improve performance. (6) Increase the number of datasets

### Provided documents

Python source code

YOLOv8n.onnx model (without providing PyTorch model)

Training map, P, R curve graph (in weights\results.png)

Test images (under test_img folder)

Please note that the dataset is provided in the dataset_address.xlsx file.


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
