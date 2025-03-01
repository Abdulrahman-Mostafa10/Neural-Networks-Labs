# Pattern Recognition and Neural Networks Lab 2

**Cairo University - Faculty of Engineering - Computer Engineering Department**  
**CMPN450 - Fall 2019**

---

## 🎨 **Lab 2: Classification Methods for Hand-Drawn Shapes**

Welcome to **Lab 2** of the _Pattern Recognition and Neural Networks_ course! In this lab, we’ll dive into the fascinating world of **machine learning** by building a system to classify hand-drawn shapes like rectangles, circles, triangles, ellipses, and polygons.

This lab will guide you through the **end-to-end pipeline** of a machine learning project, from data collection to performance evaluation. Let’s get started!

---

## 🛠️ **Machine Learning Pipeline**

### 1. **Gathering Data** 📊

The first step in any machine learning project is **data collection**. For this lab, we’ll gather images of hand-drawn shapes. The quality and quantity of the data will directly impact the performance of our model.

**Examples of shapes:**

- Rectangle
- Circle
- Triangle

---

### 2. **Data Preparation** 🧹

Raw data is often messy and needs to be cleaned before analysis. For our images, we’ll:

- Convert **RGB images to grayscale**.
- Apply a **Gaussian filter** or **low-pass filter** to smooth the images and remove noise.
- Use **thresholding** to classify pixels as black or white based on their intensity.

---

### 3. **Feature Extraction** 🔍

Feature extraction is the process of transforming raw data into meaningful features for modeling. Here are some approaches:

#### **Feature Engineering**

- **Black vs. White Pixels:** Count the number of black and white pixels in each image. (Is this a powerful feature? Can you think of better ones?)

#### **Feature Transformation**

- **Convex Hull & Bounding Shapes:**
  - Compute the **convex hull** of each shape.
  - Compare it with the **minimum enclosing circle**, **rectangle**, and **triangle**.
  - Calculate the **ratio** of the area of the shape to the area of each bounding figure.

#### **Feature Vector**

Each image will be represented as a **3D feature vector** (x, y, z), where:

- **x**: Ratio of the shape’s area to the bounding rectangle.
- **y**: Ratio of the shape’s area to the bounding circle.
- **z**: Ratio of the shape’s area to the bounding triangle.

**Example:**  
A shape with ratios (0.67, 0.58, 0.92) can be represented as the feature vector `(0.67, 0.58, 0.92)`.

---

### 4. **Model Selection** 🤖

Once we have our features, it’s time to choose a classification algorithm. We’ll experiment with three classifiers:

1. **Minimum Distance Classifier**
2. **Nearest Neighbour Classifier**
3. **K-Nearest Neighbour (K-NN) Classifier**

The goal is to compare their performance and select the best one for our task.

---

### 5. **Performance Evaluation** 📈

After training the classifier, we’ll test its accuracy on **unseen data**. The accuracy is calculated as:

\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \times 100
\]

---

## 🚀 **Key Takeaways**

- **Data is King:** The quality and quantity of your data directly impact your model’s performance.
- **Feature Engineering Matters:** Choosing the right features can make or break your model.
- **Experiment with Models:** Don’t settle for the first algorithm you try. Compare multiple models to find the best fit.

---

## 💡 **Challenge**

Can you think of additional features that could improve the classification accuracy? How would you modify the pipeline to handle more complex shapes?

---

## 📂 **Files in This Lab**

- `F19-Pattern-Lab2.pdf`: The original lab document.
- `images/`: Folder containing hand-drawn shape images.
- `test/`: Folder containing the training-images for the classifiers
- `classiciation.ipynb`: File containing the ML pipeline as required.
- `classification_utils.ipynb`: Implementation of all utilities needed.
