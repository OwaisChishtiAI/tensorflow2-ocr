** Download model files and dataset (if needed) and place into same directory as train_ocr_model.py and ocr_handwriting.py files are. **
Link: https://drive.google.com/drive/folders/1JoLYTIZmOUpkjeZXu4jO_sVYseMYTJKv?usp=sharing

For more detials go thorough these links,

For trainig:
    https://www.pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/
For Inference: 
    https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/

Inference Command:
>>> python ocr_handwriting.py --model handwriting.model --image images/umbc_zipcode.png

Training Command:
>>> python train_ocr_model.py --az a_z_handwritten_data.csv --model handwriting.model