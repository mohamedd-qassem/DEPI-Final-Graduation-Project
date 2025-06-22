# AI-Powered Early Breast Cancer Detection System 
## Our Team InnovCare Graduation Project From DEPI Scholarship

##  Overview

This is a machine learning-based web application and Mobile app designed to assist in the **early detection of breast cancer** by classifying tumors as benign or malignant. It applies **class-balanced ML models** to tackle real-world challenges like data imbalance, delivering fast, accurate, and explainable results.

 **Live App**: [Lovable Project](https://lovable.dev/projects/9d542c4e-2ce7-43d7-8e3d-8a8bdab77799)

---

##  Objectives

- Enable early prediction of breast cancer using AI.
- Handle real-world class imbalance 
- Provide clear and interpretable outputs to medical professionals.

---

##  Tech Stack

| Technology       | Role                                   |
|------------------|----------------------------------------|
| **Vite**         | Frontend build tool                    |
| **TypeScript**   | Strongly typed JavaScript              |
| **React**        | Frontend framework                     |
| **Tailwind CSS** | Utility-first styling                  |
| **shadcn/ui**    | Elegant UI components                  |
| **scikit-learn** | Machine learning models & evaluation   |
| **Lovable.dev**  | Hosting and AI-assisted development    |

---

##  Exploratory Data Analysis (EDA)

Key EDA steps included:

- **Seaborn Heatmaps** and **Matplotlib Plots** to explore:
  - Feature correlations
  - Distribution of target classes (benign vs malignant)
  - Feature behavior and variance

These steps helped identify which features were most predictive of malignancy and revealed class imbalance in the dataset.

---

##  Models Used

Two supervised learning algorithms were implemented with **class weighting** to address imbalance:

1. **Logistic Regression**
   - Suitable for binary classification
   - `class_weight='balanced'` automatically adjusts weights inversely to class frequencies

2. **Random Forest Classifier**
   - Robust against overfitting
   - Supports class weighting for skewed datasets
  
3. **Decision Tree Classifier**
   - Simple and interpretable model.
   - Useful for understanding decision boundaries and feature importance.

4. **K-Nearest Neighbors (KNN)**
   - Instance-based learning algorithm.
   - Predicts based on proximity to training examples.

all models were trained and evaluated using a traditional train/test split approach.

---

## Evaluation Metrics

To assess performance:

-  `classification_report` (includes precision, recall, f1-score)
-  `recall_score` (critical for minimizing false negatives in cancer detection)
-  Confusion Matrix (to visualize true/false positive/negatives)

Special attention was paid to **recall** to reduce the risk of missing malignant cases.

---

##  Handling Class Imbalance

This project uses:
```python
class_weight='balanced'
```

```bash
git clone git@github.com:mohamedd-qassem/DEPI-Final-Graduation-Project.git
cd DEPI-Final-Graduation-Project
npm install
npm run dev
```

##  SQL Database Integration

In addition to model training, this project includes integration with a **SQL database** to store and manage  Patient records.

 the system can persist prediction results or intermediate data into a structured database for future reference or integration with a hospital's EMR system.

This lays the groundwork for deploying the model in production environments where data logging and traceability are important.


##  Conclusion

This project highlights the strengths and trade-offs of various classifiers in early breast cancer detection:

- **Logistic Regression** and **Random Forest** performed well with class balancing.
- **Decision Trees** provided interpretability.
- **KNN** added a distance-based perspective on classification.

The final analysis suggests that **balanced class handling** and **recall-oriented evaluation** are crucial for reliable cancer detection tools.

