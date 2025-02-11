import face_recognition
import cv2
import numpy as np
from datetime import datetime
import csv
import os

# Initialize webcam
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Specifying backend for Windows

# Load images and encode faces
tesla_image = face_recognition.load_image_file(r"C:\Users\peetl\OneDrive\students1\tesla.jpg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

selva_image = face_recognition.load_image_file(r"C:\Users\peetl\OneDrive\students1\selva.jpg")
selva_encoding = face_recognition.face_encodings(selva_image)[0]

siva_image = face_recognition.load_image_file(r"C:\Users\peetl\OneDrive\students1\siva.jpg")
siva_encoding = face_recognition.face_encodings(siva_image)[0]

eswar_image = face_recognition.load_image_file(r"C:\Users\peetl\OneDrive\students1\eswar.jpg")
eswar_encoding = face_recognition.face_encodings(eswar_image)[0]

revanth_image = face_recognition.load_image_file(r"C:\Users\peetl\OneDrive\students1\revanth.jpg")
revanth_encoding = face_recognition.face_encodings(revanth_image)[0]

ashok_image = face_recognition.load_image_file(r"C:\Users\peetl\OneDrive\students1\ashok.jpg")
ashok_encoding = face_recognition.face_encodings(ashok_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [tesla_encoding, eswar_encoding, revanth_encoding, ashok_encoding]
known_face_names = ["Tesla", "Eswar", "Revanth", "Ashok"]

# Initialize variables
face_locations = []
face_encodings = []

# Get the current date and time for file naming
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = f"{current_date}.csv"

# Check if the CSV file exists, and if not, create it with headers
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Entry Time"])  # Add headers if new file

# Create a set to track recorded faces to avoid multiple entries for the same person
logged_faces = set()

while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Only log the entry if the person hasn't been recorded already
            if name not in logged_faces:
                logged_faces.add(name)

                # Get the current time
                current_time = datetime.now().strftime("%H:%M:%S")

                # Write the name and time to the CSV file
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, current_time])

                # Display the name on the webcam feed
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)  # Blue color in BGR
                thickness = 3
                lineType = 2

                cv2.putText(frame, f"{name} Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

    # Display the resulting frame
    cv2.imshow("camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()