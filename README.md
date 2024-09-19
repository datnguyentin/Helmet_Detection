# Helmet Detection and Vietnamese License Plate Detection with improved YOLOv8
---------------------------------------------------
### Demo Video
...

----------------------------------------------------
### Usage
----------------------------------------------------

### 🌟 Architecture Details 
In this project, besides vanilla YOLOv8s we also implemented attention mechanisms with two different structure
![2attentionpos](https://github.com/user-attachments/assets/445206b4-d87d-42f2-941e-cde30cf0d830)

---------------------------------------------------
### ⚖ Results and Pretrained Weights 
Pretrained Weights: [link](https://drive.google.com/drive/folders/1m8zH3VebDRmuKXfMzLrmtCr6gbXYSned?usp=sharing) \\
B: Attention modules integrated after backbone layers before concatenating (left) \\
N: Attention modules integrated before detection layers (right)

| Models          | Image size    | Params(M)| mAP@0.5| mAP@0.5:0.95 |
| -------------   |:-------------:| --------:|-------:|-------------:|
| YOLOv8s         | 640x640       | 11.1     |0.972   | 0.984        |
| -------------   |:-------------:| --------:|-------:|-------------:|
| YOLOv8s_SA      | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_ECA     | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_CBAM    | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_Triplet | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_GAM     | 640x640       | 11.1     |0.972   | 0.984        |
| -------------   |:-------------:| --------:|-------:|-------------:|
| YOLOv8s_SA      | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_ECA     | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_CBAM    | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_Triplet | 640x640       | 11.1     |0.972   | 0.984        |
| YOLOv8s_GAM     | 640x640       | 11.1     |0.972   | 0.984        |

---------------------------------------------------
### ![analysis](https://github.com/user-attachments/assets/de756475-5233-4f18-88c0-479ed287062d) Dataset
Full dataset is provided on Roboflow:
Helmet dataset: [link](https://universe.roboflow.com/datne/helmet_detection-jmhzi/dataset/3)
License_Plate dataset: [link]()
Sample Images

---------------------------------------------------
### Credits
