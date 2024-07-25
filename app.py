import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import utlis

widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [1, 1, 3, 1, 2]

# Variable to switch between webcam and static image
use_webcam = st.sidebar.checkbox("Use Webcam", value=True)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.count = 0

    def transform(self, frame):
        if use_webcam:
            img = frame.to_ndarray(format="bgr24")
        else:
            return np.zeros((heightImg, widthImg, 3), dtype=np.uint8)

        # Preprocessing
        img = cv2.resize(img, (widthImg, heightImg))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 10, 50)

        try:
            contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rectCon = utlis.rectContour(contours)
            biggestContour = utlis.getCornerPoints(rectCon[0])
            gradePoints = utlis.getCornerPoints(rectCon[1])

            if biggestContour.size != 0 and gradePoints.size != 0:
                biggestContour = utlis.reorder(biggestContour)
                gradePoints = utlis.reorder(gradePoints)

                pt1 = np.float32(biggestContour)
                pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                matrix = cv2.getPerspectiveTransform(pt1, pt2)
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

                ptg1 = np.float32(gradePoints)
                ptg2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
                matrix0 = cv2.getPerspectiveTransform(ptg1, ptg2)
                imgGradeDisplay = cv2.warpPerspective(img, matrix0, (325, 150))
                imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
                imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

                boxes = utlis.splitBoxes(imgThresh)
                myPixelVal = np.zeros((questions, choices))
                countR = 0
                countC = 0
                for image in boxes:
                    totalPixels = cv2.countNonZero(image)
                    myPixelVal[countR][countC] = totalPixels
                    countC += 1
                    if countC == choices:
                        countR += 1
                        countC = 0

                myIndex = []
                for i in range(0, questions):
                    arr = myPixelVal[i]
                    myIndexVal = np.where(arr == np.amax(arr))
                    myIndex.append(myIndexVal[0][0])

                grading = []
                for x in range(0, questions):
                    if ans[x] == myIndex[x]:
                        grading.append(1)
                    else:
                        grading.append(0)

                score = (sum(grading) / questions) * 100
                imgResult = imgWarpColored.copy()
                imgResult = utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
                imgRawDrawing = np.zeros_like(imgWarpColored)
                imgRawDrawing = utlis.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)

                invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
                imgInvWarpColored = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

                imgRawGrade = np.zeros_like(imgGradeDisplay)
                cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)

                invmatrix0 = cv2.getPerspectiveTransform(ptg2, ptg1)
                imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invmatrix0, (widthImg, heightImg))

                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarpColored, 1, 0)
                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

            else:
                st.warning("Contours not found or invalid image. Please upload a different image.")
                return np.zeros((heightImg, widthImg, 3), dtype=np.uint8)

        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
            return np.zeros((heightImg, widthImg, 3), dtype=np.uint8)

        return imgFinal

st.title("OMR MCQ Automated Grading")

if use_webcam:
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
else:
    image_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    if image_file:
        try:
            # Read and decode the image file
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None and img.size != 0:
                transformer = VideoTransformer()
                result_img = transformer.transform(img)
                st.image(result_img, caption="Result Image", use_column_width=True)
            else:
                st.error("Invalid image or image could not be loaded.")
        except Exception as e:
            st.error(f"An error occurred while reading the image: {str(e)}")
