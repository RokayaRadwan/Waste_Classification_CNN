# **Waste Classification with Custom CNN & ResNet18**

This project is a deep learning-based waste classification system using PyTorch. It classifies waste images into 20 different categories using a custom-built Convolutional Neural Network (CNN) and a pre-trained ResNet-18 model.

## **Dataset after Data Preprocessing**

- 20 classes  
- 500 images per class  
- Total: ~10,000 labeled images  

### ⚠️ **Dataset Not Included**

Due to size constraints, the dataset is **not included** in this repository.

### 📥 **How to Prepare the Dataset**

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)  
2. Remove the following classes:
   - cardboard_packaging  
   - clothing  
   - coffee_grounds  
   - food_waste  
   - magazines  
   - newspaper  
   - office_paper  
   - shoes  
   - steel_food_cans  
   - tea_bags  

## **Models**

### **1. Custom CNN**

A 5-layer convolutional model with:
- ReLU activation  
- Max pooling  
- Adaptive average pooling  
- Dropout regularization  
- Two fully connected (dense) layers  
- Weights initialization using He (Kaiming) and Xavier methods  

### **2. ResNet-18**

- A pre-trained ResNet-18 model from `torchvision.models`  
- The final fully connected layer is replaced and fine-tuned for the 20-class waste classification task  

## **Training**

The training uses:
- **CrossEntropyLoss** as the loss function  
- **Adam optimizer**  
- **Early stopping** to prevent overfitting  

### **To Train**

```bash
# Clone the repo
git clone https://github.com/yourusername/waste-classification.git
cd waste-classification

# Install dependencies
pip install torch torchvision matplotlib


## Running the Project

This project is implemented in a Jupyter Notebook.

If you're using **VS Code**:

1. Make sure you have the **Jupyter extension** installed.
2. Open the `.ipynb` file (`Waste_Classification Custom cnn model + Rasnet 18 model.ipynb`) in VS Code.
3. Run the cells using the `Run` button at the top or `Shift + Enter`.
