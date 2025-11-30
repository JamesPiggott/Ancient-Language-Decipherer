import cv2
import numpy as np
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout

# List to store coordinates of user-defined area in image to remove.
selected_area_vertices = []

class ImageProcessingApp(App):

    def build(self):
        self.load_image()
        self.scale_image()
        self.convert_to_grayscale()

        # Create a Kivy layout to contain the image and buttons
        layout = BoxLayout(orientation='vertical')

        # Display the image using a Kivy Image widget
        self.image_widget = Image(source='temp_image.jpg')
        layout.add_widget(self.image_widget)

        # Add buttons for user interaction
        mark_button = Button(text='Mark Area', on_press=self.mark_area)
        save_button = Button(text='Save Image', on_press=self.save_image)
        finish_button = Button(text='Finish', on_press=self.finish)

        button_layout = BoxLayout(orientation='horizontal')
        button_layout.add_widget(mark_button)
        button_layout.add_widget(save_button)
        button_layout.add_widget(finish_button)

        layout.add_widget(button_layout)

        return layout

    def load_image(self):
        self.orig_img = cv2.imread("../examples/sample_hieroglyphs.jpg")
        cv2.imwrite("temp_image.jpg", self.orig_img)

    def scale_image(self):
        # Scale the image to ensure it is 1000 pixels in width while maintaining its aspect ratio
        img_height, img_width, img_channels = self.orig_img.shape
        scale = 1000 / img_width
        width = int(img_width * scale)
        height = int(img_height * scale)
        self.scaled_img = cv2.resize(self.orig_img, (width, height), interpolation=cv2.INTER_AREA)

    def convert_to_grayscale(self):
        self.grey_img = cv2.cvtColor(self.scaled_img, cv2.COLOR_BGR2GRAY)

    def outline_area_callback(self, event, x, y, flags, param):
        """
        Mouse callback function to mark and save the coordinates of where the user has clicked on the image.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Mark point where user has clicked.
            cv2.circle(self.drawing_img, (x, y), 4, (0, 0, 0), -1)
            # Save click point x,y coordinates as a tuple into a list.
            selected_area_vertices.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            # Clear coordinates in current selection to allow user to restart selection.
            self.clear_selected_area_vertices()
            print("Current selection cleared. Start again.")

    def mark_area(self, instance):
        # Create a window and bind the mouse callback function to it.
        cv2.namedWindow("Area Selection")
        cv2.setMouseCallback("Area Selection", self.outline_area_callback)

        # Create an image on which to mark out areas.
        self.drawing_img = self.grey_img.copy()

        # Use the mouse click callback function to mark out areas that are not of interest in the image.
        while True:
            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == 32:
                break
            cv2.imshow("Area Selection", self.drawing_img)

        # Draw marked out area and mark the image as being modified.
        self.draw_fill_area(self.grey_img, selected_area_vertices)

        # Update the displayed image
        self.update_image()

    def save_image(self, instance):
        # Check whether the user wishes to save the image with marked out areas.
        question_save_image = "Save the modified image? (y/n): "
        if self.ask_user(question_save_image):
            # Save image.
            cv2.imwrite("area_of_interest.jpg", self.grey_img)

    def finish(self, instance):
        # Close the OpenCV window and exit
        cv2.destroyAllWindows()

    def update_image(self):
        # Update the displayed image
        cv2.imwrite("temp_image.jpg", self.grey_img)
        self.image_widget.reload()

if __name__ == "__main__":
    ImageProcessingApp().run()
