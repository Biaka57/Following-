import os
import pickle
from datetime import datetime

import time
import numpy as np
import cv2
import cvzone

import face_recognition
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import storage



cred = credentials.Certificate("resource/DB/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendance-system-77a58-default-rtdb.firebaseio.com/"
})

bucket = storage.bucket('attendance-system-77a58.appspot.com')

# access webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# load the background image
imgBackground = cv2.imread('background/background.png')

# importing  a mode into a list
folderModePath = 'resource/mode'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))

# a tv frame size
x1, y1 = 120, 150
x2, y2 = 744, 542

# background size
x3, y3 = 830, 40
x4, y4 = 1240, 690

for i in range(len(imgModeList)):
    imgModeList[i] = cv2.resize(imgModeList[i], (x4 - x3, y4 - y3))
# Load the encoding file
# print("Loading encoding file")
file = open("EncodeFile.p", 'rb')
encodeListKnownwithIds = pickle.load(file)
file.close()
encodeListKnown, stdIds = encodeListKnownwithIds
print("loaded encoded file")
print(stdIds)

modeType = 0
counter = 0
id = -1
imgStudent = []


while True:
    success, img = cap.read()

    # resize the webcam video to match the size of the TV region
    img_resized = cv2.resize(img, (x2 - x1, y2 - y1))


    # resize the image of webcam
    # imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    # imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # search encode face of known face
    faceCurFrame = face_recognition.face_locations(img_resized)
    encodeCurFrame = face_recognition.face_encodings(img_resized, faceCurFrame)

    # overlay the resized webcam video on top of the TV region
    imgBackground[y1:y2, x1:x2] = img_resized
    imgBackground[y3:y4, x3:x4] = imgModeList[modeType]

    if faceCurFrame:
        # matching encode img with face in current frame
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print("matches", matches)
            # print("Distance", faceDis)

            matchIndex = np.argmin(faceDis)
            # print(matchIndex)

            if matches[matchIndex]:

                # y1, x2, y2, x1 = faceLoc
                # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 *4, x1 * 4
                # bbox = 120+x1, 150+y1, x2 - x1, y2 - y1
                # imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                # bbox = [x1, y1, x2 - x1, y2 - y1]
                # cornerRect = cvzone.cornerRect(bbox, [x1, y1], 20, 2, (0, 255, 0), 3)
                # imgBackground = img_resized.copy()
                # imgBackground = cvzone.cornerRect(imgBackground, cornerRect, rt=0)

                id = stdIds[matchIndex]
                    # print("Known face detected..")
                    # print("faceID", stdIds[matchIndex])

                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Attendance System", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter != 0:

            if counter == 1:
                stdInfo = db.reference(f'Students/{id}').get()
                print(stdInfo)

                # get image from DB
                blob = bucket.get_blob(f'Resized/{id}.jpg')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                datatimeObject = datetime.strptime(stdInfo['last_attendance_time'], '%Y-%m-%d %H:%M:%S')
                # update last attendance time
                secondsElapsed = (datetime.now() - datatimeObject).total_seconds()
                print(secondsElapsed)

                # every after 30 seconds this can be change base on requirement
                if secondsElapsed > 30:
                    # update attendance data
                    ref = db.reference(f'Students/{id}')
                    stdInfo['total_class'] += 1
                    ref.child('total_class').set(stdInfo['total_class'])
                    stdInfo['total_present'] += 1
                    ref.child('total_present').set(stdInfo['total_present'])
                    ref.child('last_attendance_time').set(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                else:
                    modeType = 3
                    counter = 0
                    imgBackground[y3:y4, x3:x4] = imgModeList[modeType]

            if modeType != 3:
                if 10 < counter < 20:
                    modeType = 2

                imgBackground[y3:y4, x3:x4] = imgModeList[modeType]

                if counter <= 10:
                    cv2.putText(imgBackground, str(stdInfo['total_class']), (955, 82),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(imgBackground, str(stdInfo['department']), (1026, 410),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
                    cv2.putText(imgBackground, str(stdInfo['semester']), (1000, 495),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
                    cv2.putText(imgBackground, str(stdInfo['registration number']), (997, 580),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(imgBackground, str(stdInfo['total_present']), (940, 658),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(imgBackground, str(stdInfo['total_absent']), (1056, 658),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(imgBackground, str(stdInfo['year']), (1160, 658),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

                    (w, h), _ = cv2.getTextSize(stdInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 2)
                    offset = (410 - w) // 2
                    cv2.putText(imgBackground, str(stdInfo['name']), (830 + offset, 330),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                    # display std image into the frame
                    imgBackground[127:127 + 159, 955:955 + 157] = imgStudent

                counter += 1

                # reset all to zero and display mode[2] i.e Marked image
                if counter >= 20:
                    counter = 0
                    modeType = 0
                    stdInfo = []
                    imgStudent = []
                    imgBackground[y3:y4, x3:x4] = imgModeList[modeType]

    else:

        modeType = 0
        counter = 0

    for id in stdIds:
        ref = db.reference(f'Students/{id}')
        last_attendance_time = datetime.strptime(ref.child('last_attendance_time').get(),
                                                 '%Y-%m-%d %H:%M:%S')
        time_diff = (datetime.now() - last_attendance_time).total_seconds()

        if time_diff > 30:
            total_absences = ref.child('total_absent').get()
            ref.child('total_absent').set(total_absences + 1)
            total_class = ref.child('total_class').get()
            ref.child('total_class').set(total_class + 1)


        # show the result
    cv2.imshow("Attendance System", imgBackground)
    cv2.waitKey(1)
