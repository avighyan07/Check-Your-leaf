import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt

def process_image(img):
    img = cv2.resize(img, (512, 512))

# Convert image to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define a mask for GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)

    # Define the background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define the rectangle that contains the foreground object (the leaf)
    rect = (20,20, 470,470)

    # Apply GrabCut algorithm
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Extract the foreground object
    img_cut = img * mask2[:, :, np.newaxis]

    # Convert extracted image to RGB for display
    img_cut_rgb = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)

    # Display original and extracted images
    

    # Convert extracted image back to BGR
    extracted = img_cut

    # Convert to HSV color space
    hsv = cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV)

    # Define color ranges
    # Red
    l_red1 = np.array([0, 50, 50])
    u_red1 = np.array([10, 255, 255])
    l_red2 = np.array([170, 50, 50])
    u_red2 = np.array([180, 255, 255])

    # Green
    l_green = np.array([35, 50, 50])
    u_green = np.array([85, 255, 255])

    # Yellow
    l_yellow = np.array([20, 50, 50])
    u_yellow = np.array([30, 255, 255])

    # Orange
    l_orange = np.array([10, 50, 50])
    u_orange = np.array([20, 255, 255])

    # Brown
    l_brown = np.array([10, 100, 20])
    u_brown = np.array([20, 255, 200])


    # Create masks for different colors
    mask_green = cv2.inRange(hsv, l_green, u_green)
    green = cv2.bitwise_and(extracted, extracted, mask=mask_green)
    mask_yellow = cv2.inRange(hsv, l_yellow, u_yellow)
    yellow = cv2.bitwise_and(extracted, extracted, mask=mask_yellow)
    mask_brown = cv2.inRange(hsv, l_brown, u_brown)
    brown = cv2.bitwise_and(extracted, extracted, mask=mask_brown)

    mask_red1 = cv2.inRange(hsv, l_red1,u_red1 )
    red1 = cv2.bitwise_and(extracted, extracted, mask=mask_red1)
    mask_red2 = cv2.inRange(hsv, l_red2,u_red2 )
    red2 = cv2.bitwise_and(extracted, extracted, mask=mask_red2)
    mask_orange = cv2.inRange(hsv, l_orange,u_orange )
    orange = cv2.bitwise_and(extracted, extracted, mask=mask_orange)


    # Combine brown and yellow masks
    mask_brownoryellow = cv2.bitwise_or(mask_yellow, mask_brown)
    brownoryellow = cv2.bitwise_and(extracted, extracted, mask=mask_brownoryellow)
    #Combine brown, orange and yellow masks
    mask_br_y_or = cv2.bitwise_or(mask_brownoryellow,mask_orange)
    br_y_or = cv2.bitwise_and(extracted, extracted, mask=mask_br_y_or)
    # Combine  red1,green, brown,orange and yellow masks
    mask_br_y_or_r1 = cv2.bitwise_or(mask_br_y_or,mask_red1)
    br_y_or_r1 = cv2.bitwise_and(extracted, extracted, mask=mask_br_y_or_r1)
    # Combine  red1,red2,green, brown,orange and yellow masks
    mask_br_y_or_r1_r2 = cv2.bitwise_or(mask_br_y_or_r1,mask_red2)
    br_y_or_r1_r2 = cv2.bitwise_and(extracted, extracted, mask=mask_br_y_or_r1_r2)

    # Combine green, brown,orange and yellow masks
    leaf_mask = cv2.bitwise_or(mask_green, mask_br_y_or_r1_r2)
    leaf = cv2.bitwise_and(extracted, extracted, mask=leaf_mask)

    # Display color masks and extracted colors
    

    # Calculate and print pixel counts
    total_pixels = img.shape[0] * img.shape[1]
    green_pixels = np.sum(mask_green == 255)
    yellow_pixels = np.sum(mask_yellow == 255)
    brown_pixels = np.sum(mask_brown == 255)
    brownyellow_pixels = np.sum(mask_brownoryellow == 255)
    red1_pixels = np.sum(mask_red1== 255)
    red2_pixels = np.sum(mask_red2== 255)
    orange_pixels = np.sum(mask_orange ==255 )
    leaf_pixels = np.sum(leaf_mask == 255)

    diseased_pixels=yellow_pixels+brown_pixels+brownyellow_pixels+red1_pixels+red2_pixels+orange_pixels
    

    # Calculate and print disease percentage
    percentage = (diseased_pixels / leaf_pixels) * 100
    return img,leaf_mask,percentage

def main():
    st.title("Leaf Disease Detection")

    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        original_img, processed_img, diseased_percentage = process_image(image)
        
        st.subheader(f"Diseased Area Percentage: {diseased_percentage:.2f}%")
        
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)
        st.image(processed_img, caption='Processed Image', use_column_width=True)
    
if __name__ == "__main__":
    main()
