This repository contains a Random Forest classification model used to classify different categories of plant diseases. The project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, visualization, and feature importance analysis.

Project Overview
The primary objective of this model is to classify multiple plant diseases such as:

Sooty Mould
Healthy
Cutting Weevil
Gall Midge
Anthracnose
Die Back
Bacterial Canker
Powdery Mildew
The Random Forest algorithm was selected for its robustness and high accuracy in handling complex classification problems. The model has been evaluated using key metrics like accuracy, precision, recall, and F1-score.

Table of Contents
Installation
Data
Model Workflow
Results
Visualizations
Usage
Contributing
License
Installation
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/plant-disease-classification.git
cd plant-disease-classification
Install dependencies: Make sure you have Python 3.x installed. Install the required packages using:

bash
Copy code
pip install -r requirements.txt
Required Libraries:

pandas
scikit-learn
matplotlib
seaborn
numpy
graphviz (for visualizing the decision tree)
Data
The dataset used for this model contains images of leaves labeled according to different disease types. If you have your own dataset, ensure it is cleaned and properly labeled before use.

Sample features in the dataset:

Feature 1: Leaf size / color attributes
Feature 2: Weather data (optional)
Feature 3: Disease symptoms
You can modify the data loading logic in data_preprocessing.py if you are working with a different dataset format.

Model Workflow
Data Preprocessing:

Handle missing values
Label encoding for categorical variables
Train-test split (e.g., 80-20 split)
Training the Random Forest Classifier:

Parameters: n_estimators=100, max_depth=10, random_state=42
Feature importance extraction
Model Evaluation:

Accuracy, precision, recall, F1-score
Confusion matrix for better understanding of predictions
Visualization:

Decision tree plot
Top 10 feature importances
Results
Metric	Value
Accuracy	87%

Visualizations
Decision Tree Visualization: The decision tree helps understand how the model splits the data at different levels. Below is a snapshot of the complete tree visualization:

Feature Importance: Top 10 important features identified by the Random Forest model:

Usage
Training the Model
You can train the Random Forest model by running the script:

python train_model.py
Model Evaluation
Evaluate the trained model using:

python evaluate_model.py
Visualize Decision Tree
To visualize the decision tree:

python visualize_tree.py
Contributing
We welcome contributions to improve this project. To contribute:

Fork this repository.
Create a new branch (feature/your-feature).
Make your changes and commit them.
Push to the branch.
Submit a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Special thanks to the data contributors and open-source community. This project uses:

Scikit-learn for machine learning algorithms
Matplotlib and Seaborn for visualizations
Graphviz for decision tree plotting
