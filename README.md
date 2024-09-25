# Helmet Detection and Vietnamese License Plate Detection with improved YOLOv8
---------------------------------------------------
### Demo Video


https://github.com/user-attachments/assets/79cdfc21-e2b5-4050-a751-f7ff99d38bd4


----------------------------------------------------
### ![programming](https://github.com/user-attachments/assets/a2dc427f-1a99-4483-853a-f53c870be75d) Usage
In order to run the project, please install the requirements

```
pip install -r requirements.txt
```

Then move to the repository folder, run app using streamlit

```
streamlit run app.py
```

----------------------------------------------------

### 🌟 Architecture Details 
In this project, besides vanilla YOLOv8s we also implemented attention mechanisms with two different structure
![2attentionpos](https://github.com/user-attachments/assets/445206b4-d87d-42f2-941e-cde30cf0d830)

---------------------------------------------------
### ⚖ Results and Pretrained Weights 
Pretrained Weights: [link](https://drive.google.com/drive/folders/1m8zH3VebDRmuKXfMzLrmtCr6gbXYSned?usp=sharing)<br />
**B**: Attention modules integrated after backbone layers before concatenating (left).<br />


| Models            | Image size    | Params(M)| mAP@0.5| mAP@0.5:0.95 | FPS |
| -----------------:|:-------------:| --------:|-------:|-------------:|----:|
| YOLOv8s           | 640x640       | 11.1     |0.972   | 0.860        | 71  |
| YOLOv8s_SA_N      | 640x640       | 11.1     |0.974   | 0.858        | 65  |
| YOLOv8s_ECA_N     | 640x640       | 13.8     |0.972   | 0.857        | 61  |
| YOLOv8s_CBAM_N    | 640x640       | 11.6     |0.973   | 0.858        | 63  |
| YOLOv8s_Triplet_N | 640x640       | 11.1     |0.975   | 0.864        | 68  |
| YOLOv8s_GAM_N     | 640x640       | 11.1     |0.97`   | 0.856        | 68  |

**N**: Attention modules integrated before detection layers (right).

| Models            | Image size    | Params(M)| mAP@0.5| mAP@0.5:0.95 | FPS |
| -----------------:|:-------------:| --------:|-------:|-------------:|----:|
| YOLOv8s           | 640x640       | 11.1     |0.972   | 0.860        | 71  |
| YOLOv8s_SA_B      | 640x640       | 12.8     |0.970   | 0.851        | 62  |
| YOLOv8s_GAM_B     | 640x640       | 13.8     |0.970   | 0.852        | 59  |
| YOLOv8s_CBAM_B    | 640x640       | 13.0     |0.969   | 0.847        | 59  |
| YOLOv8s_ECA_B     | 640x640       | 12.8     |0.973   | 0.861        | 65  |
| YOLOv8s_Triplet_B | 640x640       | 12.8     |0.972   | 0.854        | 64  |

---------------------------------------------------
### ![analysis](https://github.com/user-attachments/assets/de756475-5233-4f18-88c0-479ed287062d) Dataset
Full dataset is provided on Roboflow:<br />
Helmet dataset: [link](https://universe.roboflow.com/datne/helmet_detection-jmhzi/dataset/3)<br />
License_Plate dataset: [link](https://universe.roboflow.com/dataset-qwb4q/license_plate_recognition-ubpod/dataset/1)<br />

Sample Images:<br />
![image](https://github.com/user-attachments/assets/edbb0b15-acfd-4859-8150-2c67058ee633)

Image in dataset<br />
![image](https://github.com/user-attachments/assets/1dc70c76-0d52-4233-b977-93113be92baa)
Image with labeled<br />

---------------------------------------------------
### ![award](https://github.com/user-attachments/assets/7b2c5bcc-044b-4cb4-af26-f760b4176782) Credits
+ Ultralytics Team: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
+ Streamlit for UI: [https://github.com/streamlit/streamlit](https://github.com/streamlit/streamlit)
