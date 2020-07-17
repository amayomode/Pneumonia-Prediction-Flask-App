# Pneumonia-Prediction-Flask-App
This is a Flask web app designed to analyze a chest x-ray and predict whether a person has TB/pneumonia or not.

The model is based on a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) that has been trained on a dataset of 800 images from two sources

  1.[Shenzhen, China](https://lhncbc.nlm.nih.gov/publication/pub9931)

  2.[Montgomery, USA](https://lhncbc.nlm.nih.gov/publication/pub9931)

The model has an overall accuracy of 83% and an F1 score of 80%.

A negative prediction means that the chest X-ray is most likely normal while the contrary is implied by a positive prediction

The web app can be found [here](https://tb-pneumonia-xray-detector.herokuapp.com/)
