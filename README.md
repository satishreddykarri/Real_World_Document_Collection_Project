# Real_World_Document_Collection_Project
# **Document Classification Using OCR**

This project performs **document classification** using Optical Character Recognition (**OCR**) to extract text from images. The model classifies documents into predefined categories such as **invoices**, **receipts**, and **reports**.

## **Project Overview**

This repository contains the steps and code for building a machine learning model that classifies documents based on their content. The approach involves:
1. **Extracting text from images using Tesseract OCR**.
2. **Cleaning and preprocessing the extracted text**.
3. **Training a classification model (Logistic Regression)** to categorize the documents.
4. **Testing the trained model** on new data.
5. **Saving the trained model and TF-IDF vectorizer** for future use.

## **Steps Covered**

### 1. **Data Collection**
The dataset is organized into labeled folders, where each folder contains images of documents in a particular category:
- **invoice**
- **receipt**
- **report**

### 2. **Text Extraction Using OCR**
- The project uses **Tesseract OCR** to extract text from images of documents in the dataset.

### 3. **Text Preprocessing**
The extracted text is preprocessed by:
- Removing unwanted characters and special symbols.
- Converting the text to lowercase.
- Removing stopwords (common, non-informative words like "the", "is", etc.).

### 4. **Feature Extraction (TF-IDF)**
The text data is converted into numerical features using **TF-IDF (Term Frequency - Inverse Document Frequency)** vectorization to represent the importance of words in the text.

### 5. **Model Training**
- A **Logistic Regression** model is trained on the vectorized text data to classify documents into categories.
- The model is evaluated on a separate test set using **accuracy** and **classification report**.

### 6. **Model Evaluation and Testing**
- The model is tested using both **text inputs** and **images**. For image inputs, Tesseract is used to extract text before classification.

### 7. **Model and Vectorizer Saving**
- After training, the **model** and **TF-IDF vectorizer** are saved for future use. This allows you to avoid retraining the model every time.

### 8. **Deployment (Future Work)**
The model can be deployed as a **REST API** using **FastAPI** to allow real-time classification of documents via text or image uploads.

## **Getting Started**

To run this project locally, follow these instructions.

### **Requirements**
1. Python 3.x
2. Tesseract OCR
3. Required Python libraries:
   - **pytesseract**
   - **pandas**
   - **numpy**
   - **scikit-learn**
   - **joblib**
   - **pillow**
   - **easyocr** (optional for faster OCR)
   - **fastapi** (for API deployment)

### **Installation**

1. **Install Tesseract OCR**:
   - **Windows**: [Download Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - **MacOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

2. **Install Python libraries**:

   You can install the required libraries using pip:
   ```bash
   pip install pytesseract pandas numpy scikit-learn joblib pillow easyocr fastapi
