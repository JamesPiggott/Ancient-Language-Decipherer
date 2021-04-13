import cv2
import numpy as np
import matplotlib.pyplot as plt


# Global variables.
# List to store coordinates of user defined area in image to remove.
selected_area_vertices = []


class ImageProcessing:

    def __init__(self):
        self.load_image()
        self.scale_image()
        self.convert_to_greyscale()
        self.modify_and_copy_image()

    def load_image(self):
        self.orig_img = cv2.imread("../examples/sample_hieroglyphs.jpg")
        cv2.imshow("Original", self.orig_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def scale_image(self):
        """
        Scale the image to ensure it is 1000 pixels in width while maintaining its aspect ratio
        """
        img_height, img_width, img_channels = self.orig_img.shape
        scale = 1000 / img_width
        width = int(img_width * scale)
        height = int(img_height * scale)
        self.scaled_img = cv2.resize(self.orig_img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Scaled", self.scaled_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_to_greyscale(self):
        """
        Convert image to greyscale.
        """
        self.grey_img = cv2.cvtColor(self.scaled_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grey", self.grey_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#     # Mouse callback function to mark and save the coordinates of where the user has clicked on the image.
#     def outline_area_callback(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             # Mark point where user has clicked.
#             cv2.circle(drawing_img, (x, y), 4, (0, 0, 0), -1)
#             # Save click point x,y coordinates as a tuple into a list.
#             selected_area_vertices.append((x, y))
#         if event == cv2.EVENT_RBUTTONDOWN:
#             # Clear coordinates in current selection to allow user to restart selection.
#             clear_selected_area_vertices()
#             print("Current selection cleared. Start again.")
#
#
#     # Function to ask user a yes/no question.
#     def ask_user(question):
#         while True:
#             response = input(question).lower()
#             if response in ["y", "n"]:
#                 # Valid response provided.
#                 if response == "y":
#                     return True
#                 else:
#                     return False

    def draw_fill_area(self, image, vertices_list):
        """
        Function to draw and fill an area using the coordinates of vertices chosen by the user through their mouse clicks.
        """
        # If there are three or more vertices in the list, draw and fill area.
        if len(vertices_list) > 2:
            cv2.fillPoly(image, np.array([vertices_list], np.int32), (0, 0, 0))
        else:
            print("Cannot draw area. Minimum of three points required.")
        # Clear coordinates list of what has been drawn or couldn't be drawn due to insufficient points.
        self.clear_selected_area_vertices()
#
#
#     # Function to clear selected area vertices list.
    def clear_selected_area_vertices(self):
        selected_area_vertices.clear()
#
#
#     # On change function for trackbar.
#     def custom_on_change(x):
#         pass
#

    """
    Mouse input functions
    """

    def outline_area_draw_rectangle(self):
        print()
        # import cv2
        # import cv2.cv as cv
        # from time import time
        # boxes = []
        #
        # def on_mouse(event, x, y, flags, params):
        #     # global img
        #     t = time()
        #
        #     if event == cv.CV_EVENT_LBUTTONDOWN:
        #         print
        #         'Start Mouse Position: ' + str(x) + ', ' + str(y)
        #         sbox = [x, y]
        #         boxes.append(sbox)
        #         # print count
        #         # print sbox
        #
        #     elif event == cv.CV_EVENT_LBUTTONUP:
        #         print
        #         'End Mouse Position: ' + str(x) + ', ' + str(y)
        #         ebox = [x, y]
        #         boxes.append(ebox)
        #         print
        #         boxes
        #         crop = img[boxes[-2][1]:boxes[-1][1], boxes[-2][0]:boxes[-1][0]]
        #
        #         cv2.imshow('crop', crop)
        #         k = cv2.waitKey(0)
        #         if ord('r') == k:
        #             cv2.imwrite('Crop' + str(t) + '.jpg', crop)
        #             print
        #             "Written to file"
        #
        # count = 0
        # while (1):
        #     count += 1
        #     img = cv2.imread('path.img', 0)
        #     # img = cv2.blur(img, (3,3))
        #     img = cv2.resize(img, None, fx=0.25, fy=0.25)

    def outline_area_callback(self, event, x, y, flags, param):
        """
        Mouse callback function to mark and save the coordinates of where the user has clicked on the image.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Testing Left Mousebutton!")
            # Mark point where user has clicked.
            cv2.circle(self.drawing_img, (x, y), 4, (0, 0, 0), -1)
            # Save click point x,y coordinates as a tuple into a list.
            selected_area_vertices.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            # Clear coordinates in current selection to allow user to restart selection.
            self.clear_selected_area_vertices()
            print("Current selection cleared. Start again.")

    def ask_user(self, question):
        """
        Function to ask user a yes/no question.
        """
        while True:
            response = input(question).lower()
            if response in ["y", "n"]:
                # Valid response provided.
                if response == "y":
                    return True
                else:
                    return False

    def modify_and_copy_image(self):
        # Create a window and bind the mouse callback function to it.
        cv2.namedWindow("Area Selection")
        cv2.setMouseCallback("Area Selection", self.outline_area_callback)
        # Create an image on which to mark out areas.
        self.drawing_img = self.grey_img.copy()
        # Use the mouse click callback function to mark out areas that are not of interest in the image.
        mark_out = True
        image_modified = False
        question_mark_out = "Mark out a region to remove from the image? (y/n): "
        question_save_image = "Save the modified image? (y/n): "
        while mark_out:
            # Check whether the user wishes to mark out an area.
            if self.ask_user(question_mark_out) is False:
                # User chose not to mark out area.
                mark_out = False
                if image_modified is True:
                    # Check whether the user wishes to save the image with marked out areas.
                    if self.ask_user(question_save_image) is True:
                        # Save image.
                        cv2.imwrite("area_of_interest.jpg", self.grey_img)
            else:
                # User chose to mark out area.
                while True:
                    # Wait 10ms for the spacebar key (ASCII code 32) to be pressed. If pressed break out of loop.
                    key_pressed = cv2.waitKey(10) & 0xFF
                    if key_pressed == 32:
                        break
                    cv2.imshow("Area Selection", self.drawing_img)
                # Draw marked out area and mark the image as being modified.
                self.draw_fill_area(self.grey_img, selected_area_vertices)
                image_modified = True
        cv2.imshow("Area Of Interest", self.grey_img)

# # Due to the nature of how light interacts with carvings and how the shadows fall, the edges of hieroglyphs in images
# # can be both light and dark. To obtain useful contours or Canny edges the majority of a hieroglyph edge must uniform.
# # Create a mask for the image that allows all colours through except the high intensity whiter range and use it to
# # replace those light edges in the same image with black pixels.
# light_mask = cv2.inRange(grey_img, 0, 210)
# mask_applied_img = cv2.bitwise_and(grey_img, grey_img, mask=light_mask)
# # cv2.imshow("Light Mask", light_mask)
# # cv2.imshow("Mask Applied", mask_applied_image)
#
# # Apply Gaussian blur to reduce noise in the image.
# blurred_img = cv2.GaussianBlur(mask_applied_img, (5, 5), 0)
# # cv2.imshow("Blurred", blurred_img)
#
# # Apply adaptive thresholding. Use inv thresholding function to make hieroglyphs in foreground white which is desired by
# # morphological transformations.
# # thresh1_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
# thresh2_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
# # cv2.imshow("adaptive_mean", thresh1_img)
# cv2.imshow("adaptive_gauss", thresh2_img)
#
# # # Use Otsu's thresholding to establish an upper and lower threshold value for Canny edge detection. It works best on
# # # bimodal images where the foreground is distinct from the background.
# # ret3, thresh3_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # ret4, thresh4_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # ret5, thresh5_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
# # ret6, thresh6_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
# # ret7, thresh7_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
# # cv2.imshow("thresh_binary_otsu", thresh3_img)
# # cv2.imshow("thresh_binary_inv_otsu", thresh4_img)
# # cv2.imshow("thresh_trunc", thresh5_img)
# # cv2.imshow("thresh_tozero_otsu", thresh6_img)
# # cv2.imshow("thresh_tozero_inv_otsu", thresh7_img)
# # lower_threshold = ret3 * 0.5  # Use for Canny edge if histogram method not used.
# # upper_threshold = ret3  # Use for Canny edge if histogram method not used.
#
# # # Apply morphological transformation to the binary image created by thresholding.
# # kernel = np.ones((3, 3), np.uint8)
# # erosion_img = cv2.erode(thresh1_img, kernel, iterations=1)
# # dilation_img = cv2.dilate(thresh1_img, kernel, iterations=1)
# # opening_img = cv2.morphologyEx(thresh1_img, cv2.MORPH_OPEN, kernel)
# # closing_img = cv2.morphologyEx(thresh1_img, cv2.MORPH_CLOSE, kernel)
# # gradient_img = cv2.morphologyEx(thresh1_img, cv2.MORPH_GRADIENT, kernel)
# # cv2.imshow("erosion", erosion_img)
# # cv2.imshow("dilation", dilation_img)
# # cv2.imshow("opening", opening_img)
# # cv2.imshow("closing", closing_img)
# # cv2.imshow("gradient", gradient_img)
#
# # Find contours on the threshold image and draw them onto a copy of the scaled image.
# contours_thresh, hierarchy_thresh = cv2.findContours(thresh2_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# threshold_contours_img = scaled_img.copy()
# cv2.drawContours(threshold_contours_img, contours_thresh, -1, (255, 0, 0), 1)
# cv2.imshow("Threshold Contours", threshold_contours_img)
#
# # Link suggests using median of image histogram to provide threshold values for Canny edge detection:
# # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
# # http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
# # Create image histogram.
# histogram = cv2.calcHist([blurred_img], [0], None, [256], [0, 256])
# plt.plot(histogram)
# plt.show()
# # Find the pixel value (x-axis of histogram) associated with the median count value. Convert histogram ndarray to a
# # list and create a list for the histogram bins which represent pixel values 0-255.
# counts = [count for [count] in histogram]
# pixel_values = list(range(0, 256))
# # Combine lists so count values are stored with their associated pixel values and sort it by counts in ascending order.
# counts_values_combined = sorted(zip(counts, pixel_values))
# median_value_location = len(counts_values_combined) // 2
# # Tuples in counts_values_combined list are structured (count, pixel value).
# median_pixel_value = counts_values_combined[median_value_location][1]
# # Calculate lower and upper threshold for Canny edge detection based on z-scores (0.66 and 1.33) which are the number
# # of standard deviations from the mean (or in this case applied to the median as it is not as affected by extremes).
# lower_threshold = 0.66 * median_pixel_value
# upper_threshold = 1.33 * median_pixel_value
#
# # Apply Canny edge detection.
# edges_img = cv2.Canny(blurred_img, lower_threshold, upper_threshold, apertureSize=3)
# cv2.imshow("Canny", edges_img)
#
# # Find contours on the Canny image and draw them onto a copy of the scaled image.
# contours_canny, hierarchy_canny = cv2.findContours(edges_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# canny_contours_img = scaled_img.copy()
# cv2.drawContours(canny_contours_img, contours_canny, -1, (0, 0, 255), 1)
# cv2.imshow("Canny Contours", canny_contours_img)
#
# # Find the Hough lines. Create a window to hold the trackbars and image.
# cv2.namedWindow("Hough")
# # Create trackbars that can be used to adjust Hough transform parameters.
# cv2.createTrackbar("min_line_length", "Hough", 150, 300, custom_on_change)
# cv2.createTrackbar("max_line_gap", "Hough", 150, 300, custom_on_change)
# cv2.createTrackbar("threshold", "Hough", 150, 300, custom_on_change)
# # Create a copy of the scaled image onto which the Hough lines will be drawn.
# hough_lines_img = scaled_img.copy()
#
# while True:
#     # Wait 10ms for the ESC key (ASCII code 27) to be pressed. If pressed break out of loop.
#     key_pressed = cv2.waitKey(10) & 0xFF
#     if key_pressed == 27:
#         break
#
#     # Return position of each trackbar.
#     min_line_length = cv2.getTrackbarPos("min_line_length", "Hough")
#     max_line_gap = cv2.getTrackbarPos("max_line_gap", "Hough")
#     threshold = cv2.getTrackbarPos("threshold", "Hough")
#
#     # Find/highlight the long horizontal and vertical lines that bound the hieroglyphs in the image by applying the
#     # probabilistic Hough Transform (unlike standard Hough it uses only a random subset of the points so is less
#     # computationally intensive). May then be possible to isolate these regions of interest.
#     lines = cv2.HoughLinesP(thresh2_img, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=min_line_length,
#                             maxLineGap=max_line_gap)
#
#     # Plot only the horizontal and vertical Hough lines (if there are any) on a copy of the scaled colour image. With
#     # each loop, the Hough lines image is reset to a clean scaled image with no lines on it before plotting again.
#     # Lines are unlikely to be exactly horizontal/vertical (i.e. x1 != x2 and y1 != y2) but are assumed to be if within
#     # a tolerance value (in pixels). If x1 and x2 are within tolerance the line is considered vertical. If y1 and y2 are
#     # within tolerance the line is considered horizontal.
#     hough_lines_img = scaled_img.copy()
#     if lines is not None:
#         for line in lines:
#             tolerance = 10
#             x1, y1, x2, y2 = line[0]
#             if x1 - tolerance <= x2 <= x1 + tolerance or y1 - tolerance <= y2 <= y1 + tolerance:
#                 cv2.line(hough_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     # Show Hough lines.
#     cv2.imshow("Hough", hough_lines_img)
#
# # Show final Hough lines image.
# cv2.imshow("Final Hough", hough_lines_img)
#
# # Wait for keypress then destroy all open windows.
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Start the image processing application
    """
    app = ImageProcessing()
