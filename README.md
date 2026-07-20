🌿 Leaf Disease Prediction using Deep Learning (CNN)

🚀 Live Demo

Web Application   👉 https://leaf-disease-prediction-dl-akif7.onrender.com
<img width="2928" height="1666" alt="image" src="https://github.com/user-attachments/assets/74dc1868-6462-45bc-82a0-e96e5d4a8759" />

<img width="2102" height="1640" alt="image" src="https://github.com/user-attachments/assets/288e8660-0936-4e31-9686-639ad008f4f8" />

<img width="2018" height="1670" alt="image" src="https://github.com/user-attachments/assets/9e48bdbe-2d9e-45e3-a36d-0561c700d29d" />

<img width="2072" height="1658" alt="image" src="https://github.com/user-attachments/assets/75c16fa9-c974-49e7-a871-cc6204321206" />


🎯 Features

* Upload leaf image
* Deep Learning based prediction
* Custom CNN model
* Confidence score
* Top-3 predictions
* Healthy/Diseased status
* Disease information
* Symptoms
* Causes
* Treatment
* Prevention
* Similar dataset images
* Fully Responsive UI
* Flask Backend
* Render Deployment

🌱 Supported Crops

Bell Pepper
* Healthy
* Bacterial Spot
* 
Potato
* Healthy
* Early Blight
* Late Blight

Tomato
* Healthy
* Bacterial Spot
* Early Blight
* Late Blight
* Leaf Mold
* Septoria Leaf Spot
* Spider Mites
* Target Spot
* Yellow Leaf Curl Virus
* Tomato Mosaic Virus

Total Classes 15

📸 Image Preprocessing
Before prediction, every image passes through:

* Resize to 128×128
* RGB conversion
* Convert to NumPy array
* Float32 conversion
* Normalize pixels (/255.0)
* Expand dimensions for batch prediction


🔄 Data Augmentation
Training images were augmented using TensorFlow preprocessing layers:

* Random Flip
* Random Rotation
* Random Zoom
* Random Contrast

This improves model generalization and reduces overfitting.

🛠 TensorFlow / Keras Components Used

Core Libraries
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn

TensorFlow Layers
* Input
* Conv2D
* MaxPooling2D
* BatchNormalization
* Dropout
* GlobalAveragePooling2D
* Dense
* RandomFlip
* RandomRotation
* RandomZoom
* RandomContrast

TensorFlow Utilities
* ImageDataGenerator / image_dataset_from_directory
* EarlyStopping
* ModelCheckpoint
* ReduceLROnPlateau
* Adam Optimizer
* SparseCategoricalCrossentropy

📂 Project Structure
  Leaf_Disease_Prediction

│── app.py

│── requirements.txt

│── leaf_disease_model.keras

│── dataset/

│── static/

│── templates/

│── notebooks/

│── README.md


🌐 Deployment
Platform : Render

Live URL
https://leaf-disease-prediction-dl-akif7.onrender.com


🔮 Future Improvements

* Support more crop species
* Mobile App
* Grad-CAM visualization
* Disease severity estimation
* Camera-based detection
* REST API
* Farmer dashboard
* Multi-language support


👨‍💻 Author

Mohammad Akif Akhtar
AI & Machine Learning Engineer

GitHub : https://github.com/mdakifAkhtar
LinkedIn: https://www.linkedin.com/in/mohammad-akif-akhtar-b4a6042b8?utm_source=share_via&utm_content=profile&utm_medium=member_android

Live Demo
https://leaf-disease-prediction-dl-akif7.onrender.com


⭐ Show Your Support
If you found this project useful, please ⭐ star this repository and share it with others.
