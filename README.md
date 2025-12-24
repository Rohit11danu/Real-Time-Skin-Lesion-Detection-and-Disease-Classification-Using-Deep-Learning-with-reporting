This model is used to  detect real time skin diseases.
This model is divided into three types
First  part is to detect the object means the skin disease  (we gave it a name lesian_detected) (using YOLOv8 ML algorihtm) ( it detect by pattern ,skin texture , color)
Second part is to classify the detected_lesian into disease classes (we have used 4 classes ) (for image preprocessing we use open cv) (for image classification we used Resnet 18)
Third part is to make a human redable report .

