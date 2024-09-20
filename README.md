# Helmet Detection and Vietnamese License Plate Detection with improved YOLOv8
---------------------------------------------------
### Demo Video


https://github.com/user-attachments/assets/79cdfc21-e2b5-4050-a751-f7ff99d38bd4


----------------------------------------------------
### ![programming](https://github.com/user-attachments/assets/a2dc427f-1a99-4483-853a-f53c870be75d)Usage
In order to run the project, please install the requirements

'''
pip install -r requirements.txt
'''

Then move to the repository folder, run app using streamlit

'''
streamlit run app.py
'''

----------------------------------------------------

### 🌟 Architecture Details 
In this project, besides vanilla YOLOv8s we also implemented attention mechanisms with two different structure
![2attentionpos](https://github.com/user-attachments/assets/445206b4-d87d-42f2-941e-cde30cf0d830)

---------------------------------------------------
### ⚖ Results and Pretrained Weights 
Pretrained Weights: [link](https://drive.google.com/drive/folders/1m8zH3VebDRmuKXfMzLrmtCr6gbXYSned?usp=sharing)<br />
**B**: Attention modules integrated after backbone layers before concatenating (left).<br />


| Models            | Image size    | Params(M)| mAP@0.5| mAP@0.5:0.95 |
| -----------------:|:-------------:| --------:|-------:|-------------:|
| YOLOv8s           | 640x640       | 11.1     |0.972   | 0.984        |
| -------------     |---------------| ---------|--------|--------------|
| YOLOv8s_SA_N      | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_ECA_N     | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_CBAM_N    | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_Triplet_N | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_GAM_N     | 640x640       | 11.1     |0.972   | 0.984        |
| ------------------|---------------| ---------|--------|--------------|

**N**: Attention modules integrated before detection layers (right).

| Models            | Image size    | Params(M)| mAP@0.5| mAP@0.5:0.95 |
| -----------------:|:-------------:| --------:|-------:|-------------:|
| YOLOv8s           | 640x640       | 11.1     |0.972   | 0.984        |
| -------------     |---------------| ---------|--------|--------------|
| YOLOv8s_SA_B      | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_ECA_B     | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_CBAM_B    | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_Triplet_B | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_GAM_B     | 640x640       | 11.1     |0.972   | 0.984        |
| ------------------|---------------| ---------|--------|--------------|

---------------------------------------------------
### ![analysis](https://github.com/user-attachments/assets/de756475-5233-4f18-88c0-479ed287062d) Dataset
Full dataset is provided on Roboflow:<br />
Helmet dataset: [link](https://universe.roboflow.com/datne/helmet_detection-jmhzi/dataset/3)<br />
License_Plate dataset: [link]()__
Sample Images:
![Light_1_16085](https://github.com/user-attachments/assets/3ba2c2c7-ceee-460c-837b-f595a2e62c62)
Image in dataset
![image](https://github.com/user-attachments/assets/1dc70c76-0d52-4233-b977-93113be92baa)
Image with labeled

---------------------------------------------------
### Credits
