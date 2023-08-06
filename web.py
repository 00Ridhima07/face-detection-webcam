import cv2

#Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('frontalfacedefault.xml')

#Capture webcam vid to use, 0 is for webcam, otherwise add vid file name
webcam= cv2.VideoCapture(0)

#Loop forever over frames until you end program
while True:
    #Read the current frame
    successful_frame_read, frame = webcam.read()

    #Grayscale capture (less data used, makes program run faster)
    grayscaled_vid= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #Detect Faces, detectmultiscale detects diff sizes and returns coordinates of rectangles surrounding face
    #First 2 numbers are coordinates of top left corner and second 2 are bottom right corner
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_vid)

    #Draw rectangles around the faces, (0,255,0) is BGR colour, 2 is thickness of rectangle
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)



    #Show img w faces
    cv2.imshow('Face det', frame)

    #Keeps img/frame open, without it img would close right after running
    #number is for waitkey being clicked per ms
    key = cv2.waitKey(1)

    #Stop if q is clicked
    if key==81 or key==113:
        break
#Release the videocapture object/end program 
webcam.release()
