from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

#Face detection
#Types of backends
backends= ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]
'''
#face recognition
dfs = DeepFace.find(
  img_path = "img1.jpg",
  db_path = "C:/workspace/my_db", 
  model_name = models[1],
)

#face verification
result = DeepFace.verify(
  img1_path = "D:/FutureExpertData/FaceRecognition/DeepFaceProject/people/Dominique_de_Villepin.jpg",
  img2_path = "D:/FutureExpertData/FaceRecognition/DeepFaceProject/dataset/Dominique_de_Villepin/Dominique_de_Villepin_0005_rotation.png",
  model_name = models[0],
)
print(result)
'''
#Face extraction
alignment_modes = [True, False]



fig, axs = plt.subplots(3, 2, figsize=(15, 10))
axs = axs.flatten()
for i, b in enumerate(backends):
    #face detection and alignment
    try:
        face_objs = DeepFace.extract_faces(
        img_path = "D:/FutureExpertData/FaceRecognition/DeepFaceProject/people/Dominique_de_Villepin.jpg", 
        detector_backend = b,
        align = alignment_modes[0],
        ) 
        axs[i].imshow(face_objs[0]['face'])
    except:
        pass
plt.show()

plt.title("Extracted Face")
plt.axis('off')  # Hide axes for a cleaner look
plt.show()