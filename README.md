# Vehicle Classification Model for Prospect Auto

## Project Overview

In this project, I developed a machine learning model to classify vehicles based on their silhouettes for **Prospect Auto**, a chain of car repair shops. The goal was to create a classification model that could predict the vehicle class with high accuracy, based on silhouette features. To solve this problem, I explored multiple machine learning algorithms and evaluated their performance using various metrics.

## Problem Statement

Prospect Auto needed a solution to classify vehicles based on their silhouettes. The goal was to build a model that could accurately predict the vehicle class. This project involves data preprocessing, model training, evaluation, and final recommendations.

## My Approach

I followed the typical machine learning project pipeline, with a focus on:
1. **Data Preprocessing**: I loaded and cleaned the dataset, performed exploratory data analysis (EDA), and prepared the data for modeling. This included normalizing and splitting the data into training and testing sets.
2. **Model Testing**: I tested multiple machine learning algorithms, including:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM) with different kernels (Linear, Radial Basis Function (RBF), and Polynomial)
   - Decision Tree
   - Random Forest
   - Gradient Boosting

3. **Model Evaluation**: I evaluated each model based on various metrics, such as:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC AUC Score

4. **Recommendation**: After testing and evaluating the models, I compared the results using various visualizations, including a heatmap and line plots.

## Key Findings

Through the evaluation, I found that the **Support Vector Machine (SVM) with the Radial Basis Function (RBF) kernel** performed the best across all metrics. The key findings include:
- **SVM (RBF Kernel)** achieved scores greater than 99% in all the key metrics: Accuracy, Precision, Recall, F1-score, and ROC AUC score.
- The **line plot** further confirmed this, showing a consistent and near-ideal performance on both the training and testing data.
- When comparing the train-test ratio, the SVM with RBF kernel maintained a ratio close to 1 across all metrics, indicating excellent generalization from training to testing.

## My Recommendation

Based on the results, I recommend that **Prospect Auto** use the **Support Vector Machine (SVM) with the Radial Basis Function (RBF) kernel** for vehicle classification. This model demonstrated the highest performance, making it the best choice for this task.

## Steps Taken in the Project

1. **Data Preprocessing**:  
   - Loaded and cleaned the dataset.
   - Performed exploratory data analysis (EDA) to understand the structure and distribution of the data.
   - Normalized and standardized the dataset to ensure effective training of machine learning models.

2. **Model Training & Testing**:  
   - Trained and tested the following models:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**
     - **Support Vector Machine (SVM)** with Linear, RBF, and Polynomial kernels
     - **Decision Tree**
     - **Random Forest**
     - **Gradient Boosting**

3. **Model Evaluation**:  
   - Evaluated each model using key metrics such as Accuracy, Precision, Recall, F1-Score, and ROC AUC score.
   - Used visualizations like heatmaps and line plots to compare model performance.

4. **Final Model Selection**:  
   - Based on performance, I chose the **SVM with RBF kernel** as the best model due to its superior results across all metrics.

## Visualizations

The following visualizations were created to compare the performance of different models:
- **Heatmap**: Showcasing the key metrics for each model.
- **Line Plot**: Displaying the train-test ratio and performance consistency for each model.

These visualizations provide a clear overview of model performance and help in making an informed decision about which model to deploy.

## Libraries Used

The following libraries were used in this project:
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib** and **seaborn**: For data visualization and exploratory analysis.
- **scikit-learn**: For machine learning algorithms, model evaluation, and preprocessing.
- **IPython**: For displaying the results and visualizations.
- **pprint**: For pretty-printing data structures.
- **time**: For measuring computation times.

## Conclusion

The **Support Vector Machine (SVM) with RBF kernel** provides the best performance for vehicle classification in this project, and I highly recommend it for use by **Prospect Auto**. This model's accuracy and ability to generalize well make it a reliable choice for real-world deployment.

## Link to the Notebook

You can access the complete project in my [Supervised-ML-Notebook](https://colab.research.google.com/drive/1Tpzphy-Iz6-DIkPIXkzoCup-rVywpp2u?usp=sharing).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
