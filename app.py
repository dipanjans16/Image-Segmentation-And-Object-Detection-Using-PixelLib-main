# import necessary libraries
import cv2 # computer vision library
import pixellib # pixel-level segmentation library
from pixellib.instance import instance_segmentation # import instance segmentation model

# initialize instance segmentation model and load pre-trained model weights
segmenter = instance_segmentation()
segmenter.load_model("mask_rcnn_coco.h5") 

# open the video capture device
vid_capture = cv2.VideoCapture(0)

# start the video capture loop
while vid_capture.isOpened():
    # read a frame from the video capture device
    ret, frame = vid_capture.read()
    
    # apply instance segmentation on the frame to identify objects and create segmentation mask
    result = segmenter.segmentFrame(frame, show_bboxes=True)
    segmented_image = result[1] # retrieve the segmented image from the result
    
    # display the segmented image on a window
    cv2.imshow('Image Segmentation', segmented_image)

    # wait for user input to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# release the video capture device and destroy all open windows
vid_capture.release()
cv2.destroyAllWindows()
