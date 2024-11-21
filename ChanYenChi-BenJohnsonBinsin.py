import tkinter as tk
from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import filedialog, colorchooser, messagebox
from PIL import Image, ImageOps, ImageTk, ImageFilter, ImageFont, ImageDraw
from tkinter import ttk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
            
class MainInterface:

    def __init__(self):
        
        self.root = tk.Tk()
        self.root.geometry("1100x700")
        self.root.title("Image Editor")
        self.root.config(bg="gray")

        self.pen_color = "black"
        self.pen_size = 5
        self.file_path = ""
        self.cropping = False
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
        self.original_image = None
        self.edited_image = None
        self.crosshair_size = 10
        self.text_color = "black"
        self.text_x, self.text_y = 0, 0
        self.text_font = ImageFont.load_default()
        self.text_content = ""
        self.text_size = 50
        self.text_font = ImageFont.load_default()
        self.drawing_enabled = False
        self.text_dragging = False
        self.text_x_offset = 0
        self.rotation_angle = 0
        self.r1 = 70
        self.s1 = 0
        self.r2 = 140
        self.s2 = 255
        self.createWidgets()
        

    def add_image(self):
        file_path = filedialog.askopenfilename(initialdir="C:/Users/ASUS/Pictures")
        if file_path:
            self.file_path = file_path  # Update the file path here
            new_image = cv2.imread(file_path)

            # Display the original image in the main window if not already displayed
            if self.original_image is None:
                self.original_image = new_image
                self.display_image(self.original_image)
            else:
                # Create a new temporary window for the newly loaded image
                temp_window = tk.Toplevel(self.root)
                temp_window.title("New Image Viewer")

                # Convert image from BGR to RGB for displaying with tkinter
                new_image_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
                new_image_pil = Image.fromarray(new_image_rgb)

                # Display the new image in the temporary window
                temp_canvas = tk.Canvas(temp_window, width=new_image_pil.width, height=new_image_pil.height)
                temp_canvas.pack()

                new_image_tk = ImageTk.PhotoImage(new_image_pil)
                temp_canvas.create_image(0, 0, image=new_image_tk, anchor="nw")
                temp_canvas.image = new_image_tk
                
    def open_image(self):
        file_path = filedialog.askopenfilename(initialdir="C:/Users/ASUS/Pictures")
        if file_path:
            self.file_path = file_path  # Update the file path here
            new_image = Image.open(file_path)
            self.original_image = new_image
            self.display_open_image(self.original_image)


    def display_open_image(self, image, text="", text_color="black", text_x=0, text_y=0, font_size=12):
        image_tk = ImageTk.PhotoImage(image)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, image=image_tk, anchor="nw")
        self.canvas.image = image_tk
        self.edited_image = image.copy()

        if text:
            text_font = ("Arial", font_size)
            self.canvas.create_text(text_x, text_y, text=text, fill=text_color, anchor="nw", font=text_font)


    def display_image(self, image, text="", text_color="black", text_x=0, text_y=0, font_size=12):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.canvas.config(width=image_pil.width, height=image_pil.height)
        self.canvas.create_image(0, 0, image=image_tk, anchor="nw")
        self.canvas.image = image_tk
        self.edited_image = image.copy()

        if text:
            text_font = ("Arial", font_size)
            self.canvas.create_text(text_x, text_y, text=text, fill=text_color, anchor="nw", font=text_font)

    def change_edit_image(self):
        new_file_path = filedialog.askopenfilename(initialdir="C:/Users/ASUS/Pictures")
        if new_file_path:
            self.original_image = cv2.imread(new_file_path)
            self.display_image(self.original_image)
            
    def toggle_drawing(self):
        self.drawing_enabled = not self.drawing_enabled

    def change_color(self):
        color = colorchooser.askcolor(title="Select Pen Color")[1]
        self.pen_color = color

    def change_size(self, size):
        self.pen_size = size

    def draw(self, event):
        if self.drawing_enabled:
            x1, y1 = (event.x - self.pen_size), (event.y - self.pen_size)
            x2, y2 = (event.x + self.pen_size), (event.y + self.pen_size)
            self.canvas.create_oval(x1, y1, x2, y2, fill=self.pen_color, outline='')

    def change_text_color(self):
        color = colorchooser.askcolor(title="Select Text Color")[1]
        self.text_color = color
        self.display_open_image(self.original_image, text=self.text_entry.get(), text_color=self.text_color,
                           text_x=self.text_x, text_y=self.text_y)

    def change_text_size(self, size):
        self.text_size = size
        self.display_open_image(self.original_image, text=self.text_entry.get(), text_color=self.text_color,
                           text_x=self.text_x, text_y=self.text_y, font_size=size)

    def start_text_drag(self, event):
        if self.original_image:
            self.text_dragging = True
            self.text_x_offset = event.x - self.text_x
            self.text_y_offset = event.y - self.text_y

    def drag_text(self, event):
        if self.text_dragging and self.original_image:
            self.text_x = event.x - self.text_x_offset
            self.text_y = event.y - self.text_y_offset
            self.display_open_image(self.original_image, text=self.text_entry.get(), text_color=self.text_color,
                               text_x=self.text_x, text_y=self.text_y, font_size=self.text_size)

    def end_text_drag(self, event):
        self.text_dragging = False

    def toggle_text_drag(self):
        if self.text_drag_button["text"] == "Enable Text Drag":
            self.text_drag_button["text"] = "Disable Text Drag"
            self.canvas.bind("<ButtonPress-1>", self.start_text_drag)
            self.canvas.bind("<B1-Motion>", self.drag_text)
            self.canvas.bind("<ButtonRelease-1>", self.end_text_drag)
        else:
            self.text_drag_button["text"] = "Enable Text Drag"
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
        
    def draw_circle(self):
        self.drawing_enabled = True
        if self.original_image:
            self.canvas.bind("<ButtonPress-1>", self.start_draw_circle)

    def start_draw_circle(self, event):
        self.x_start, self.y_start = event.x, event.y
        self.canvas.bind("<B1-Motion>", self.draw_circle_motion)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw_circle)

    def draw_circle_motion(self, event):
        self.canvas.delete("temp_shape")
        self.canvas.create_oval(self.x_start, self.y_start, event.x, event.y, outline="red", width=2, tag="temp_shape")

    def end_draw_circle(self, event):
        self.canvas.delete("temp_shape")
        if self.original_image:
            draw = ImageDraw.Draw(self.original_image)
            draw.ellipse([(self.x_start, self.y_start), (event.x, event.y)], outline="red", width=2)
            self.display_open_image(self.original_image)
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.drawing_enabled = False

    def draw_rectangle(self):
        self.drawing_enabled = True
        if self.original_image:
            self.canvas.bind("<ButtonPress-1>", self.start_draw_rectangle)

    def start_draw_rectangle(self, event):
        self.x_start, self.y_start = event.x, event.y
        self.canvas.bind("<B1-Motion>", self.draw_rectangle_motion)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw_rectangle)

    def draw_rectangle_motion(self, event):
        self.canvas.delete("temp_shape")
        self.canvas.create_rectangle(self.x_start, self.y_start, event.x, event.y, outline="blue", width=2, tag="temp_shape")

    def end_draw_rectangle(self, event):
        self.canvas.delete("temp_shape")
        if self.original_image:
            draw = ImageDraw.Draw(self.original_image)
            draw.rectangle([(self.x_start, self.y_start), (event.x, event.y)], outline="blue", width=2)
            self.display_open_image(self.original_image)
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.drawing_enabled = False

    def draw_line(self):
        self.drawing_enabled = True
        if self.original_image:
            self.canvas.bind("<ButtonPress-1>", self.start_draw_line)

    def start_draw_line(self, event):
        self.x_start, self.y_start = event.x, event.y
        self.canvas.bind("<B1-Motion>", self.draw_line_motion)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw_line)

    def draw_line_motion(self, event):
        self.canvas.delete("temp_shape")
        self.canvas.create_line(self.x_start, self.y_start, event.x, event.y, fill="orange", width=2, tag="temp_shape")

    def end_draw_line(self, event):
        self.canvas.delete("temp_shape")
        if self.original_image:
            draw = ImageDraw.Draw(self.original_image)
            draw.line([(self.x_start, self.y_start), (event.x, event.y)], fill="orange", width=2)
            self.display_open_image(self.original_image)
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.drawing_enabled = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor="nw")

    def apply_filter(self, filter_name):
        if self.file_path and self.original_image:
            image = self.original_image.copy()  # Create a copy of the original image to apply filters
            if filter_name == "GrayScale":
                image = ImageOps.grayscale(image)
            if filter_name == "RGB":  # Convert to RGB color space
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif filter_name == "HSV":  # Convert to HSV color space
                image = ImageOps.colorize(image.convert('L'), "#00FF00", "#FF0000")
            elif filter_name == "CIE":  # Convert to CIE color space
                image = ImageOps.colorize(image.convert('L'), "#FF00FF", "#FFFF00")
            elif filter_name == "HLS":  # Convert to HLS color space
                image = ImageOps.colorize(image.convert('L'), "#FF0000", "#0000FF")
            elif filter_name == "YCrCb":  # Convert to YCrCb color space
                image = ImageOps.colorize(image.convert('L'), "#0000FF", "#FFFF00")

            self.display_open_image(image)
                
    def start_crop(self, event):
        if event.num == 3 and self.original_image:
            self.x_start, self.y_start = event.x, event.y
            self.canvas.delete("guide")
            self.canvas.create_rectangle(self.x_start, self.y_start, self.x_start, self.y_start, outline='red', tag="guide")
            self.canvas.create_line(self.x_start - self.crosshair_size, self.y_start, self.x_start + self.crosshair_size, self.y_start, fill='red')
            self.canvas.create_line(self.x_start, self.y_start - self.crosshair_size, self.x_start, self.y_start + self.crosshair_size, fill='red')
            self.cropping = True

    def update_crop(self, event):
        if self.cropping and self.original_image:
            x_end, y_end = event.x, event.y
            self.canvas.coords("guide", self.x_start, self.y_start, x_end, y_end)
            self.canvas.coords(1, self.x_start - self.crosshair_size, self.y_start, self.x_start + self.crosshair_size, self.y_start)
            self.canvas.coords(2, self.x_start, self.y_start - self.crosshair_size, self.x_start, self.y_start + self.crosshair_size)
            
    def display_cropped_image(self, cropped_img):
        cropped_window = tk.Toplevel(self.root)
        cropped_window.title("Cropped Image")
        cropped_canvas = tk.Canvas(cropped_window, width=cropped_img.width, height=cropped_img.height)
        cropped_canvas.pack()
        cropped_image_tk = ImageTk.PhotoImage(cropped_img)
        cropped_canvas.create_image(0, 0, image=cropped_image_tk, anchor="nw")
        cropped_canvas.image = cropped_image_tk

    def end_crop(self, event):
        if event.num == 3 and self.original_image:
            x_end, y_end = event.x, event.y
            self.canvas.delete("guide")
            cropped_image = self.original_image.crop((min(self.x_start, x_end), min(self.y_start, y_end),
                                                     max(self.x_start, x_end), max(self.y_start, y_end)))
            self.display_cropped_image(cropped_image)
            self.cropping = False
            
    def rotate_image(self, degrees):
        if self.original_image:
            self.original_image = self.original_image.rotate(degrees)
            self.display_open_image(self.original_image)

    def scale_image(self):
        if self.original_image:
            try:
                x_scale = float(self.x_scale_entry.get())
                y_scale = float(self.y_scale_entry.get())
                resized_image = self.original_image.resize((int(self.original_image.width * x_scale),
                                                            int(self.original_image.height * y_scale)),
                                                           Image.ANTIALIAS)
                self.display_open_image(resized_image)
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter valid scaling factors (float or integer values).")


    def translate_image(self, x_offset=0, y_offset=0):
        if self.original_image:
            try:
                x_offset = int(self.x_offset_entry.get())
                y_offset = int(self.y_offset_entry.get())
                translated_image = self.original_image.copy()
                translated_image = translated_image.transform(
                    translated_image.size, Image.AFFINE, (1, 0, -x_offset, 0, 1, y_offset))
                self.display_open_image(translated_image)
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter valid integer values for X and Y offsets.")

    def image_info(self):
        if self.original_image:
            image_format = self.original_image.format
            image_mode = self.original_image.mode
            image_size = self.original_image.size
            info_str = f"File Path: {self.file_path}\nImage Format: {image_format}\nImage Mode: {image_mode}\nImage Size: {image_size}"
            tk.messagebox.showinfo("Image Information", info_str)

    def reset_image(self):
        if self.file_path:
            self.original_image = Image.open(self.file_path)
            self.display_open_image(self.original_image)

    def save_image(self):
        if self.edited_image:
            filename = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                    filetypes=[("JPG files", "*.jpg"), ("All Files", "*.*")])
            if filename:
                cv2_image = cv2.cvtColor(np.array(self.edited_image), cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, cv2_image)
                self.file_path = filename  # Update file path to the saved imag  
                
    def split_red_channel(self):
        (b, g, r) = cv2.split(self.original_image)
        cv2.imshow('red channel',r)
        cv2.waitKey(1) 

    def split_green_channel(self):
        (b, g, r) = cv2.split(self.original_image)
        cv2.imshow('green channel',g)
        cv2.waitKey(1) 

    def split_blue_channel(self):
        (b, g, r) = cv2.split(self.original_image)
        cv2.imshow('blue channel',b)
        cv2.waitKey(1) 

    def combine_channels(self):
        (b, g, r) = cv2.split(self.original_image)
        image_merged = cv2.merge((b,g,r))
        cv2.imshow('merged image',image_merged) 
        cv2.waitKey(1) 
            
    def plot_histogram(self, channel):
        img_array = np.array(self.original_image)
        hist, _ = np.histogram(img_array[:, :, 'rgb'.index(channel)], 256, [0, 256])
        plt.plot(hist, color=channel)
        plt.xlim([0, 256])
        plt.title(f'{channel.upper()} Channel Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

    def plot_combined_histogram(self):
        img_array = np.array(self.original_image)
        red_hist, _ = np.histogram(img_array[:, :, 0], 256, [0, 256])
        green_hist, _ = np.histogram(img_array[:, :, 1], 256, [0, 256])
        blue_hist, _ = np.histogram(img_array[:, :, 2], 256, [0, 256])

        plt.figure(figsize=(8, 6))
        plt.plot(red_hist, color='r', label='Red Channel')
        plt.plot(green_hist, color='g', label='Green Channel')
        plt.plot(blue_hist, color='b', label='Blue Channel')
        plt.xlim([0, 256])
        plt.title('Histogram for RGB Channels')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
        
    def bitSlicing(self):
        bitSlicinggray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_BGR2GRAY)
        imgs = [255 * ((bitSlicinggray & (1 << i)) >> i) for i in range(8)]

        bit_plane_images = []
        for i in range(8):
            bit_plane_images.append(np.uint8(imgs[i]))

        # Display bit-plane images using cv2.imshow()
        for i in range(8):
            cv2.imshow(f'Bit Plane {i+1}', bit_plane_images[i])
            cv2.waitKey(2000)  # Display each image for 1 second
            cv2.destroyAllWindows()

        # Display bit-plane images using Matplotlib's plt.show()
        fig = plt.figure(figsize=(10, 5))
        columns = 4
        rows = 2
        for i in range(1, columns * rows + 1):
            img = bit_plane_images[i-1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap='gray')
            plt.title(f'Bit Plane {i}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        
    def cannyEdgeDetect(self):
        self.gray_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_BGR2GRAY)
        edgeCanny = cv2.Canny(self.gray_image, 30, 200)

        plt.title("Canny Edge Detection")
        plt.imshow(edgeCanny, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def prewittEdgeDetect(self):
        self.gray_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv2.filter2D(self.gray_image, -1, kernelx)
        img_prewitty = cv2.filter2D(self.gray_image, -1, kernely)
        img_prewitt = img_prewittx + img_prewitty

        plt.title("Prewitt Edge Detection")
        plt.imshow(img_prewitt, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def sobelEdgeDetect(self):
        self.gray_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

        plt.title("Sobel Edge Detection")
        plt.imshow(sobel, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    def EdgeDetection(self):
        title_window1 = 'Edge Detection - Window 1'
        title_window2 = 'Edge Detection - Window 2'

        cv2.namedWindow(title_window1)
        cv2.namedWindow(title_window2)

        # Define callback function
        def operations(val):
            pass
            
        # Create Trackbars for Window 1
        cv2.createTrackbar('Canny1', title_window1, 0, 200, operations)
        cv2.createTrackbar('Prewitt1', title_window1, 0, 3, operations)
        cv2.createTrackbar('Sobel1', title_window1, 0, 3, operations)
        cv2.createTrackbar('Robert1', title_window1, 0, 3, operations)
        cv2.createTrackbar("Mode1", title_window1, 0, 4, operations)
        
        # Create Trackbars for Window 2
        cv2.createTrackbar('Canny2', title_window2, 0, 200, operations)
        cv2.createTrackbar('Prewitt2', title_window2, 0, 3, operations)
        cv2.createTrackbar('Sobel2', title_window2, 0, 3, operations)
        cv2.createTrackbar('Robert2', title_window2, 0, 3, operations)
        cv2.createTrackbar("Mode2", title_window2, 0, 4, operations)

        while True:
            img1 = self.original_image.copy()
            img2 = self.original_image.copy()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Get trackbar values for window 1
            canny1 = int(cv2.getTrackbarPos('Canny1', title_window1))
            prewitt1 = int(cv2.getTrackbarPos('Prewitt1', title_window1))
            sobel1 = int(cv2.getTrackbarPos('Sobel1', title_window1))
            robert1 = int(cv2.getTrackbarPos('Robert1', title_window1))
            s1 = cv2.getTrackbarPos("Mode1", title_window1)

            # Get trackbar values for window 2
            canny2 = int(cv2.getTrackbarPos('Canny2', title_window2))
            prewitt2 = int(cv2.getTrackbarPos('Prewitt2', title_window2))
            sobel2 = int(cv2.getTrackbarPos('Sobel2', title_window2))
            robert2 = int(cv2.getTrackbarPos('Robert2', title_window2))
            s2 = cv2.getTrackbarPos("Mode2", title_window2)

            if s1==1:
                if canny1>0:
                    #img1=cv2.Canny(self.img, canny/2, canny/2, apertureSize=5, L2gradient = True)
                    t_lower = canny1 # Lower Threshold
                    t_upper = 200  # Upper threshold
                    aperture_size = 5  # Aperture size
                    L2Gradient = True             
                    # Applying the Canny Edge filter
                    # with Custom Aperture Size
                    img1 = cv2.Canny(self.original_image, canny1, t_upper, apertureSize=aperture_size, L2gradient = L2Gradient)
                else:
                    pass
            elif s1==2:
                if prewitt1==1:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
                    img1 = cv2.filter2D(gray, -1, kernelx)
                elif prewitt1==2:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                    img1 = cv2.filter2D(gray, -1, kernely)
                elif prewitt1==3:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
                    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                    img_prewittx = cv2.filter2D(gray, -1, kernelx)
                    img_prewitty = cv2.filter2D(gray, -1, kernely)
                    img1= img_prewittx + img_prewitty
                else:
                    pass
            elif s1==3:
                 if sobel1==1:
                    self.img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    img1 = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
                 elif sobel1==2:
                    self.img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    img1 = cv2.Sobel(self.img, cv2.CV_64F, 0, 1,ksize=5)
                 elif sobel1==3:
                    self.img= cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    img_sobelx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
                    img_sobely = cv2.Sobel(self.img, cv2.CV_64F, 0, 1,ksize=5)
                    img1 = img_sobelx + img_sobely
                 else:
                    pass
            elif s1==4:
                if robert1==1:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelX=np.array([[1,0],[0,-1]])
                    robertImageX=np.uint8(np.absolute(gray))
                    img1 = cv2.filter2D(gray, -1, kernelX)
                elif robert1==2:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelY=np.array([[0,1],[-1,0]])
                    robertImageY=np.uint8(np.absolute(gray))
                    img1 = cv2.filter2D(gray, -1, kernelY)
                elif robert1==3:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelX=np.array([[1,0],[0,-1]])
                    kernelY=np.array([[0,1],[-1,0]])
                    img_robertx = cv2.filter2D(gray, -1, kernelX)
                    img_roberty = cv2.filter2D(gray, -1, kernelY)
                    robertImage=np.uint8(np.absolute(gray))
                    img1= img_robertx + img_roberty
                else:
                    pass
            else:
                pass
            
            if s2==1:
                if canny2>0:
                    #img1=cv2.Canny(self.img, canny/2, canny/2, apertureSize=5, L2gradient = True)
                    t_lower = canny2 # Lower Threshold
                    t_upper = 200  # Upper threshold
                    aperture_size = 5  # Aperture size
                    L2Gradient = True             
                    # Applying the Canny Edge filter
                    # with Custom Aperture Size
                    img2 = cv2.Canny(self.original_image, canny2, t_upper, apertureSize=aperture_size, L2gradient = L2Gradient)
                else:
                    pass
            elif s2==2:
                if prewitt2==1:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
                    img2 = cv2.filter2D(gray, -1, kernelx)
                elif prewitt2==2:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                    img2 = cv2.filter2D(gray, -1, kernely)
                elif prewitt2==3:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
                    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                    img_prewittx = cv2.filter2D(gray, -1, kernelx)
                    img_prewitty = cv2.filter2D(gray, -1, kernely)
                    img2= img_prewittx + img_prewitty
                else:
                    pass
            elif s2==3:
                 if sobel2==1:
                    self.img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    img2 = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
                 elif sobel2==2:
                    self.img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    img2 = cv2.Sobel(self.img, cv2.CV_64F, 0, 1,ksize=5)
                 elif sobel2==3:
                    self.img= cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    img_sobelx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
                    img_sobely = cv2.Sobel(self.img, cv2.CV_64F, 0, 1,ksize=5)
                    img2 = img_sobelx + img_sobely
                 else:
                    pass
            elif s2==4:
                if robert2==1:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelX=np.array([[1,0],[0,-1]])
                    robertImageX=np.uint8(np.absolute(gray))
                    img2 = cv2.filter2D(gray, -1, kernelX)
                elif robert2==2:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelY=np.array([[0,1],[-1,0]])
                    robertImageY=np.uint8(np.absolute(gray))
                    img2 = cv2.filter2D(gray, -1, kernelY)
                elif robert2==3:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    kernelX=np.array([[1,0],[0,-1]])
                    kernelY=np.array([[0,1],[-1,0]])
                    img_robertx = cv2.filter2D(gray, -1, kernelX)
                    img_roberty = cv2.filter2D(gray, -1, kernelY)
                    robertImage=np.uint8(np.absolute(gray))
                    img2= img_robertx + img_roberty
                else:
                    pass
            else:
                pass
            
            cv2.imshow(title_window1, img1)
            cv2.imshow(title_window2, img2)
            
        cv2.destroyAllWindows()
            
    def realtimeEdge(self):
    # OpenCV program to perform Edge detection in real time 
    # import libraries of python OpenCV  
    # where its functionality resides 
    
    # capture frames from a camera 
        cap = cv2.VideoCapture(0) 
        
        # loop runs if capturing has been initialized 
        while(1): 
        
            # reads frames from a camera 
            ret, frame = cap.read() 
        
            # converting BGR to HSV 
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # define range of red color in HSV 
            lower_red = np.array([30,150,50]) 
            upper_red = np.array([255,255,180]) 
            
            # create a red HSV colour boundary and  
            # threshold HSV image 
            mask = cv2.inRange(hsv, lower_red, upper_red) 
        
            # Bitwise-AND mask and original image 
            res = cv2.bitwise_and(frame,frame, mask= mask) 
        
            # Display an original image 
            cv2.imshow('Original',frame) 
        
            # finds edges in the input image image and 
            # marks them in the output map edges 
            sobelx=cv2.Sobel(gray_frame,cv2.CV_64F,1,0)
            sobely=cv2.Sobel(gray_frame,cv2.CV_64F,0,1)
            sobelx=np.uint8(np.absolute(sobelx))
            sobely=np.uint8(np.absolute(sobely))
            sobelCombine=cv2.bitwise_or(sobelx,sobely)
            
            kernelX=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            kernelY=np.array([[-1,0,-1],[-1,0,1],[1,0,1]])
            prewittX=cv2.filter2D(gray_frame,-1,kernelX)
            prewittY=cv2.filter2D(gray_frame,-1,kernelY)
            prewittCombine=cv2.bitwise_or(prewittX,prewittY)
            
            kernelX=np.array([[1,0],[0,-1]])
            kernelY=np.array([[0,1],[-1,0]])
            robertX=cv2.filter2D(gray_frame,-1,kernelX)
            robertY=cv2.filter2D(gray_frame,-1,kernelY)
            robertImageX=np.uint8(np.absolute(robertX))
            robertImageY=np.uint8(np.absolute(robertY))
            robertCombine=cv2.bitwise_or(robertX,robertY)
            
            edgeCanny=cv2.Canny(gray_frame,30,200)
            
            # Display edges in a frame 
            cv2.imshow('Canny',edgeCanny)
            cv2.imshow('Prewitt Combine',prewittCombine)
            cv2.imshow('Robert Combine',robertCombine) 
            cv2.imshow('Sobel Combine',sobelCombine)  
        
            # Wait for Esc key to stop 
            k = cv2.waitKey(5) & 0xFF
            if k == ord('q'): 
                break
        
        # Close the window 
        cap.release() 
        
        # De-allocate any associated memory usage 
        cv2.destroyAllWindows()
        
    def countourImage(self):
        if self.original_image is not None and self.original_image.size > 0:
            cv_image = np.array(self.original_image)
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
            _, threshold = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

            # Initialize a variable for contour intensity
            contour_intensity = 2

            # Callback function for the trackbar
            def update_contour_intensity(val):
                nonlocal contour_intensity
                contour_intensity = val
                # Redraw contours when the trackbar is adjusted
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour_image = cv_image.copy()
                createContour = cv2.drawContours(contour_image, contours, -1, (0, 255, 0), contour_intensity)
                cv2.imshow('Contour Image', createContour)

            # Create a window to display both image and trackbar
            cv2.namedWindow('Contour Image')

            # Create a trackbar for adjusting contour intensity
            cv2.createTrackbar('Intensity', 'Contour Image', contour_intensity, 10, update_contour_intensity)

            while True:
                # Display initial contours
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour_image = cv_image.copy()
                createContour = cv2.drawContours(contour_image, contours, -1, (0, 255, 0), contour_intensity)
                cv2.imshow('Contour Image', createContour)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break

                # Retrieve the current intensity value from the trackbar
                contour_intensity = cv2.getTrackbarPos('Intensity', 'Contour Image')

        else:
            print("No original image loaded.")

        cv2.destroyAllWindows()

            
    def filter(self):
        # Create window titles for trackbars
        title_window1 = 'Filter 1'
        title_window2 = 'Filter 2'

        # Define callback function
        def operations(val):
            pass

        # Create Trackbars for Filter 1
        cv2.namedWindow(title_window1)
        cv2.createTrackbar('Bilateral 1', title_window1, 0, 100, operations)
        cv2.createTrackbar('Gaussian Blur 1', title_window1, 0, 11, operations)
        cv2.createTrackbar('Median Filter 1', title_window1, 0, 11, operations)
        cv2.createTrackbar('Average Filter 1', title_window1, 0, 20, operations)
        cv2.createTrackbar('Sigma 1', title_window1, 0, 4, operations)
        cv2.createTrackbar('Box Filter 1 Intensity', title_window1, 0, 10, operations)

        # Create Trackbars for Filter 2
        cv2.namedWindow(title_window2)
        cv2.createTrackbar('Bilateral 2', title_window2, 0, 100, operations)
        cv2.createTrackbar('Gaussian Blur 2', title_window2, 0, 11, operations)
        cv2.createTrackbar('Median Filter 2', title_window2, 0, 11, operations)
        cv2.createTrackbar('Average Filter 2', title_window2, 0, 20, operations)
        cv2.createTrackbar('Sigma 2', title_window2, 0, 4, operations)
        cv2.createTrackbar('Box Filter 2 Intensity', title_window2, 0, 10, operations)

        while True:
            img1 = self.original_image.copy()
            img2 = self.original_image.copy()
            
            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' key to quit
                break
            
            # Filter 1 Trackbars
            bilaterall_1 = int(cv2.getTrackbarPos('Bilateral 1', title_window1))
            gaussiann_1 = int(cv2.getTrackbarPos('Gaussian Blur 1', title_window1))
            mediann_1 = int(cv2.getTrackbarPos('Median Filter 1', title_window1))
            averagee_1 = int(cv2.getTrackbarPos('Average Filter 1', title_window1))
            gammaa_1 = int(cv2.getTrackbarPos('Sigma 1', title_window1))
            box_intensity_1 = int(cv2.getTrackbarPos('Box Filter 1 Intensity', title_window1))

            # Filter 2 Trackbars
            bilaterall_2 = int(cv2.getTrackbarPos('Bilateral 2', title_window2))
            gaussiann_2 = int(cv2.getTrackbarPos('Gaussian Blur 2', title_window2))
            mediann_2 = int(cv2.getTrackbarPos('Median Filter 2', title_window2))
            averagee_2 = int(cv2.getTrackbarPos('Average Filter 2', title_window2))
            gammaa_2 = int(cv2.getTrackbarPos('Sigma 2', title_window2))
            box_intensity_2 = int(cv2.getTrackbarPos('Box Filter 2 Intensity', title_window2))

            # Apply filters sequentially
            bilateralfilter_1 = cv2.bilateralFilter(img1, 2 * bilaterall_1 + 1, 75, 75)
            gaussianFilter_1 = cv2.GaussianBlur(bilateralfilter_1, (2 * gaussiann_1 + 1, 2 * gaussiann_1 + 1), 0)
            medianFilter_1 = cv2.medianBlur(gaussianFilter_1, 2 * mediann_1 + 1)
            blur_1 = cv2.blur(medianFilter_1, (2 * averagee_1 + 1, 2 * averagee_1 + 1))
            invGamma_1 = 1 / 2 * gammaa_1 + 1
            table_1 = [((i / 255) ** invGamma_1) * 255 for i in range(256)]
            table_1 = np.array(table_1, np.uint8)
            img1 = cv2.LUT(blur_1, table_1)
            
            bilateralfilter_2 = cv2.bilateralFilter(img2, 2 * bilaterall_2 + 1, 75, 75)
            gaussianFilter_2 = cv2.GaussianBlur(bilateralfilter_2, (2 * gaussiann_2 + 1, 2 * gaussiann_2 + 1), 0)
            medianFilter_2 = cv2.medianBlur(gaussianFilter_2, 2 * mediann_2 + 1)
            blur_2 = cv2.blur(medianFilter_2, (2 * averagee_2 + 1, 2 * averagee_2 + 1))
            invGamma_2 = 1/ 2*gammaa_2+1
            table_2 = [((i / 255) ** invGamma_2) * 255 for i in range(256)]
            table_2 = np.array(table_2, np.uint8)
            img2 = cv2.LUT(blur_2, table_2)
            
            if box_intensity_1 != 0:
                img1 = cv2.boxFilter(img1, -1, (3, 3), img1, (-1, -1), False, cv2.BORDER_DEFAULT)
                img1 = cv2.addWeighted(img1, box_intensity_1 / 10, img1, 0, 0)

            if box_intensity_2 != 0:
                img2 = cv2.boxFilter(img2, -1, (3, 3), img2, (-1, -1), False, cv2.BORDER_DEFAULT)
                img2 = cv2.addWeighted(img2, box_intensity_2 / 10, img2, 0, 0)
            
            # Display the images
            cv2.imshow(title_window1, img1)
            cv2.imshow(title_window2, img2)

        cv2.destroyAllWindows()

            
    def sharpImage(self):
        self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Initialize sharpness variable
        sharpness = 1

        # Callback function for the sharpness trackbar
        def update_sharpness_kernel1(val):
            nonlocal sharpness
            sharpness = val / 10  # Adjust the sharpness factor as needed
            redraw_sharpness_kernel1()

        def update_sharpness_kernel2(val):
            nonlocal sharpness
            sharpness = val / 10  # Adjust the sharpness factor as needed
            redraw_sharpness_kernel2()

        # Function to redraw the image with updated sharpness and kernel1
        def redraw_sharpness_kernel1():
            kernel1 = np.array([[-1, -1, -1], [-1, 9 + sharpness, -1], [-1, -1, -1]])
            sharp = cv2.filter2D(self.image, -1, kernel1)
            cv2.imshow('Sharp Image (Kernel 1)', sharp)

        # Function to redraw the image with updated sharpness and kernel2
        def redraw_sharpness_kernel2():
            kernel2 = np.array([[-1, -1, -1, -1, -1],
                                [-1, -1, -1, -1, -1],
                                [-1, -1, 20 + sharpness * 4, -1, -1],
                                [-1, -1, -1, -1, -1],
                                [-1, -1, -1, -1, -1]])
            sharp = cv2.filter2D(self.image, -1, kernel2)
            cv2.imshow('Sharp Image (Kernel 2)', sharp)

        # Create windows to display the images
        cv2.namedWindow('Sharp Image (Kernel 1)')
        cv2.namedWindow('Sharp Image (Kernel 2)')

        # Create trackbars for adjusting sharpness for each kernel
        cv2.createTrackbar('Sharpness (Kernel 1)', 'Sharp Image (Kernel 1)', int(sharpness * 10), 20, update_sharpness_kernel1)
        cv2.createTrackbar('Sharpness (Kernel 2)', 'Sharp Image (Kernel 2)', int(sharpness * 10), 20, update_sharpness_kernel2)

        # Initial image drawing
        redraw_sharpness_kernel1()
        redraw_sharpness_kernel2()

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

        cv2.destroyAllWindows()

    def powerLaw(self):
        self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        gamma_value_twopoints = 2.2  # Initial gamma value for twopoints
        gamma_value_fourpoints = 0.4  # Initial gamma value for fourpoints

        # Callback function for the gamma value trackbar (twopoints)
        def update_gamma_twopoints(val):
            nonlocal gamma_value_twopoints
            gamma_value_twopoints = val / 10  # Adjust the gamma value as needed
            apply_gamma_correction_twopoints()

        # Callback function for the gamma value trackbar (fourpoints)
        def update_gamma_fourpoints(val):
            nonlocal gamma_value_fourpoints
            gamma_value_fourpoints = val / 10  # Adjust the gamma value as needed
            apply_gamma_correction_fourpoints()

        # Function to apply gamma correction (twopoints) and display the image
        def apply_gamma_correction_twopoints():
            gamma_corrected_twopoints = np.array(255 * (self.image / 255) ** gamma_value_twopoints, dtype='uint8')
            cv2.imshow('Gamma Corrected Image (Twopoints)', gamma_corrected_twopoints)

        # Function to apply gamma correction (fourpoints) and display the image
        def apply_gamma_correction_fourpoints():
            gamma_corrected_fourpoints = np.array(255 * (self.image) ** gamma_value_fourpoints, dtype='uint8')
            cv2.imshow('Gamma Corrected Image (Fourpoints)', gamma_corrected_fourpoints)

        # Create separate windows for twopoints and fourpoints gamma correction
        cv2.namedWindow('Gamma Corrected Image (Twopoints)')
        cv2.namedWindow('Gamma Corrected Image (Fourpoints)')

        # Create trackbars for adjusting gamma values for twopoints and fourpoints
        cv2.createTrackbar('Gamma Value (Twopoints)', 'Gamma Corrected Image (Twopoints)', int(gamma_value_twopoints * 10), 40, update_gamma_twopoints)
        cv2.createTrackbar('Gamma Value (Fourpoints)', 'Gamma Corrected Image (Fourpoints)', int(gamma_value_fourpoints * 10), 40, update_gamma_fourpoints)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

        cv2.destroyAllWindows()

    def Image_dilation(self):
        
        def update_dilation(val):
            kernel_sizes = [(0,0),(3, 3), (5, 5), (7, 7), (5, 5), (7, 7)]
            iterations = [0,1, 1, 1, 2, 2]

            kernel_size = kernel_sizes[val]
            iteration = iterations[val]

            perform_operation(kernel_size, iteration)

        def perform_operation(kernel_size, iteration):
            kernel = np.ones(kernel_size, np.uint8)
            dilated_image = cv2.dilate(self.original_image, kernel, iterations=iteration)
            cv2.imshow('Dilation', dilated_image)

        cv2.namedWindow('Dilation')
        cv2.createTrackbar('Kernel', 'Dilation', 0, 5, update_dilation)
        
        update_dilation(0)  # Initial operation with default settings

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

        cv2.destroyAllWindows()
        
    def Image_erosion(self):
        def update_erosion(val):
            kernel_sizes = [(0, 0),(3, 3), (5, 5), (7, 7)]
            kernels = [np.zeros((0, 0), np.uint8),np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8),np.ones((7, 7), np.uint8)]

            kernel = kernels[val]
            perform_operation(kernel, val)

        def perform_operation(kernel, val):
            if val == 0:
                eroded_image = self.original_image
            else:
                if val == 1:
                    eroded_image = cv2.erode(self.original_image, kernel)
                else:
                    eroded_image = cv2.erode(self.original_image, kernel, borderType=cv2.BORDER_REFLECT)

            cv2.imshow('Erosion', eroded_image)

        cv2.namedWindow('Erosion')
        cv2.createTrackbar('Kernel', 'Erosion', 0, 3, update_erosion)

        update_erosion(0)  # Initial operation with default settings

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

        cv2.destroyAllWindows()

    def thresholdingImg(self):
        image_grey = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        thresh_1 = 127  # Initial threshold value for window 1
        thresh_2 = 127  # Initial threshold value for window 2
        mode_1 = 0  # Initial mode: 0 for 'THRESH_BINARY' for window 1
        mode_2 = 0  # Initial mode: 0 for 'THRESH_BINARY' for window 2

        def update_thresh_1(val):
            nonlocal thresh_1
            thresh_1 = val
            perform_thresholding(1)

        def update_thresh_2(val):
            nonlocal thresh_2
            thresh_2 = val
            perform_thresholding(2)

        def update_mode_1(val):
            nonlocal mode_1
            mode_1 = val
            perform_thresholding(1)

        def update_mode_2(val):
            nonlocal mode_2
            mode_2 = val
            perform_thresholding(2)

        def perform_thresholding(window):
            if window == 1:
                ret, thresh1 = cv2.threshold(image_grey, thresh_1, 255, cv2.THRESH_BINARY)
                ret, thresh2 = cv2.threshold(image_grey, thresh_1, 255, cv2.THRESH_BINARY_INV)
                ret, thresh3 = cv2.threshold(image_grey, thresh_1, 255, cv2.THRESH_TRUNC)
                ret, thresh4 = cv2.threshold(image_grey, thresh_1, 255, cv2.THRESH_TOZERO)
                ret, thresh5 = cv2.threshold(image_grey, thresh_1, 255, cv2.THRESH_TOZERO_INV)
                thresholds = [thresh1, thresh2, thresh3, thresh4, thresh5]
                selected_mode = mode_1
            else:
                ret, thresh1 = cv2.threshold(image_grey, thresh_2, 255, cv2.THRESH_BINARY)
                ret, thresh2 = cv2.threshold(image_grey, thresh_2, 255, cv2.THRESH_BINARY_INV)
                ret, thresh3 = cv2.threshold(image_grey, thresh_2, 255, cv2.THRESH_TRUNC)
                ret, thresh4 = cv2.threshold(image_grey, thresh_2, 255, cv2.THRESH_TOZERO)
                ret, thresh5 = cv2.threshold(image_grey, thresh_2, 255, cv2.THRESH_TOZERO_INV)
                thresholds = [thresh1, thresh2, thresh3, thresh4, thresh5]
                selected_mode = mode_2

            cv2.imshow(f'Thresholding {window}', thresholds[selected_mode])

        cv2.namedWindow('Thresholding 1')
        cv2.createTrackbar('Threshold 1', 'Thresholding 1', 0, 255, update_thresh_1)
        cv2.createTrackbar('Mode 1', 'Thresholding 1', 0, 4, update_mode_1)

        cv2.namedWindow('Thresholding 2')
        cv2.createTrackbar('Threshold 2', 'Thresholding 2', 0, 255, update_thresh_2)
        cv2.createTrackbar('Mode 2', 'Thresholding 2', 0, 4, update_mode_2)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

        cv2.destroyAllWindows()
        
    def adaptiveThresholding(self):
        image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        thresh = 127  # Initial threshold value for both windows
        mode_1 = 0  # Initial mode for window 1
        mode_2 = 0  # Initial mode for window 2

        def update_adaptive_thresh_1(val):
            nonlocal thresh
            thresh = val
            perform_adaptive_thresholding(1)

        def update_adaptive_thresh_2(val):
            nonlocal thresh
            thresh = val
            perform_adaptive_thresholding(2)

        def update_mode_1(val):
            nonlocal mode_1
            mode_1 = val
            perform_adaptive_thresholding(1)

        def update_mode_2(val):
            nonlocal mode_2
            mode_2 = val
            perform_adaptive_thresholding(2)

        def perform_adaptive_thresholding(window):
            ret, th1 = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
            th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            ret, th4 = cv2.threshold(image, thresh, 255, cv2.THRESH_OTSU)

            thresholds = [th1, th2, th3, th4]

            if window == 1:
                cv2.imshow('Adaptive Thresholding 1', thresholds[mode_1])
            else:
                cv2.imshow('Adaptive Thresholding 2', thresholds[mode_2])

        cv2.namedWindow('Adaptive Thresholding 1')
        cv2.createTrackbar('Thresh Val 1', 'Adaptive Thresholding 1', 0, 255, update_adaptive_thresh_1)
        cv2.createTrackbar('Mode 1', 'Adaptive Thresholding 1', 0, 3, update_mode_1)

        cv2.namedWindow('Adaptive Thresholding 2')
        cv2.createTrackbar('Thresh Val 2', 'Adaptive Thresholding 2', 0, 255, update_adaptive_thresh_2)
        cv2.createTrackbar('Mode 2', 'Adaptive Thresholding 2', 0, 3, update_mode_2)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

        cv2.destroyAllWindows()
        
    def EqualizeHist(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        equ = cv2.equalizeHist(gray_image)

        # Calculate histograms
        hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist_equalized = cv2.calcHist([equ], [0], None, [256], [0, 256])

        # Stacking images side-by-side
        res = np.hstack((gray_image, equ))

        # Show image input vs output
        cv2.imshow('Image', res)

        # Plot histograms
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(hist_original, color='b')
        plt.title('Original Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(122)
        plt.plot(hist_equalized, color='r')
        plt.title('Equalized Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                plt.close()
                break

    def pixelVal(self, pix, r1, s1, r2, s2): 
            if (0 <= pix and pix <= r1): 
                return (s1 / r1) * pix 
            elif (r1 < pix and pix <= r2): 
                return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1 
            else: 
                return ((255 - s2) / (255 - r2)) * (pix - r2) + s2 

    def apply_contrast_stretching(self):
        img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        pixelVal_vec = np.vectorize(self.pixelVal)
        contrast_stretched = pixelVal_vec(img, self.r1, self.s1, self.r2, self.s2)
        
        cv2.namedWindow('contrast_stretching')
        
        def update_values(*args):
            nonlocal img
            self.r1 = cv2.getTrackbarPos('r1', 'contrast_stretching')
            self.s1 = cv2.getTrackbarPos('s1', 'contrast_stretching')
            self.r2 = cv2.getTrackbarPos('r2', 'contrast_stretching')
            self.s2 = cv2.getTrackbarPos('s2', 'contrast_stretching')
            contrast_stretched = pixelVal_vec(img, self.r1, self.s1, self.r2, self.s2)
            cv2.imshow('contrast_stretching', contrast_stretched)
            
            hist = cv2.calcHist([contrast_stretched.astype('uint8')], [0], None, [256], [0, 256])
            hist = hist * 400 / hist.max()
            hist_img = np.zeros((400, 256, 3), dtype=np.uint8)
            for x, h in enumerate(hist):
                cv2.line(hist_img, (x, 400), (x, 400 - int(h)), (255, 255, 255))

            cv2.imshow('Histogram', hist_img)

        cv2.createTrackbar('r1', 'contrast_stretching', self.r1, 255, update_values)
        cv2.createTrackbar('s1', 'contrast_stretching', self.s1, 255, update_values)
        cv2.createTrackbar('r2', 'contrast_stretching', self.r2, 255, update_values)
        cv2.createTrackbar('s2', 'contrast_stretching', self.s2, 255, update_values)

        cv2.imshow('contrast_stretching', contrast_stretched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def split(self):
        img = np.array(self.original_image)  # Convert canvas to numpy array
        h, w, channels = img.shape

        half_width = w // 2
        half_height = h // 2

        left_part = img[:, :half_width]
        right_part = img[:, half_width:]
        top_part = img[:half_height, :]
        bottom_part = img[half_height:, :]

        unique_id = str(uuid.uuid4())[:8]  # Generate a unique ID
        cv2.imshow('Left part', left_part)
        cv2.imshow('Right part', right_part)
        cv2.imshow('Top part', top_part)
        cv2.imshow('Bottom part', bottom_part)

        cv2.imwrite(f'D:/UMS/Visual Studio Code/SEM5/SplittedImage/left_part_{unique_id}.jpg', left_part)
        cv2.imwrite(f'D:/UMS/Visual Studio Code/SEM5/SplittedImage/right_part_{unique_id}.jpg', right_part)
        cv2.imwrite(f'D:/UMS/Visual Studio Code/SEM5/SplittedImage/top_part_{unique_id}.jpg', top_part)
        cv2.imwrite(f'D:/UMS/Visual Studio Code/SEM5/SplittedImage/bottom_part_{unique_id}.jpg', bottom_part)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def combine_images_horizontal(self, num):
        image_paths = []
        for i in range(num):
            image_path = filedialog.askopenfilename(initialdir='D:/UMS/Visual Studio Code/SEM5/SplittedImage', title=f"Select Image {i+1}")
            if image_path:
                image_paths.append(image_path)

        if len(image_paths) == num:
            images = []
            for path in image_paths:
                image = cv2.imread(path)
                if image is not None:
                    image_resized = cv2.resize(image, (350, 300))
                    images.append(image_resized)

            frames = []
            for image in images:
                frame = np.zeros((300, 400, 3), dtype=np.uint8)
                frame[0:300, :350] = image
                frames.append(frame)

            combined_image = cv2.hconcat(frames)
            cv2.imshow('Combined Image', combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def combine_images_vertical(self, num):
        image_paths = []
        for i in range(num):
            image_path = filedialog.askopenfilename(initialdir='D:/UMS/Visual Studio Code/SEM5/SplittedImage', title=f"Select Image {i+1}")
            if image_path:
                image_paths.append(image_path)

        if len(image_paths) == num:
            images = []
            for path in image_paths:
                image = cv2.imread(path)
                if image is not None:
                    image_resized = cv2.resize(image, (350, 300))
                    images.append(image_resized)

            frames = []
            for image in images:
                frame = np.zeros((300, 400, 3), dtype=np.uint8)
                frame[0:300, :350] = image
                frames.append(frame)

            combined_image = cv2.vconcat(frames)
            cv2.imshow('Combined Image', combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def createWidgets(self):
        
        self.upper_frame = tk.Frame(self.root, height=30, bg="light gray")
        self.upper_frame.pack(side="top", fill="x")
        
        self.left_frame = tk.Frame(self.root, width=200, height=600, bg="gray")
        self.left_frame.pack(side="left", fill="y")
        
        self.right_frame = tk.Frame(self.root, width=160, height=600, bg="gray")
        self.right_frame.pack(side="right", fill="y")

        self.canvas = tk.Canvas(self.root, width=750, height=700)
        self.canvas.pack()

        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Add Image", command=self.add_image)
        self.file_menu.add_command(label="Open Image", command=self.open_image)
        self.file_menu.add_command(label="Change Edit Image", command=self.change_edit_image)
        self.file_menu.add_command(label="Image Info", command=self.image_info)
        self.file_menu.add_command(label="Reset Image", command=self.reset_image)
        self.file_menu.add_command(label="Save Image", command=self.save_image)
        
        self.filter_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Filters", menu=self.filter_menu)

        self.filter_menu.add_command(label="GrayScale", command=lambda: self.apply_filter("GrayScale"))
        self.filter_menu.add_command(label="RGB", command=lambda: self.apply_filter("RGB"))
        self.filter_menu.add_command(label="HSV", command=lambda: self.apply_filter("HSV"))
        self.filter_menu.add_command(label="CIE", command=lambda: self.apply_filter("CIE"))
        self.filter_menu.add_command(label="HLS", command=lambda: self.apply_filter("HLS"))
        self.filter_menu.add_command(label="YCrCb", command=lambda: self.apply_filter("YCrCb"))
        
        histogram_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Histogram", menu=histogram_menu)
        
        histogram_menu.add_command(label="Red Channel", command=lambda: self.plot_histogram('r'))
        histogram_menu.add_command(label="Green Channel", command=lambda: self.plot_histogram('g'))
        histogram_menu.add_command(label="Blue Channel", command=lambda: self.plot_histogram('b'))
        histogram_menu.add_command(label="All Channels", command=self.plot_combined_histogram)
        
        self.edge_detection_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edge Detection", menu=self.edge_detection_menu)

        self.edge_detection_menu.add_command(label="Edge Detection", command=self.EdgeDetection)
        self.edge_detection_menu.add_command(label="Real TIme Edge Detection", command=self.realtimeEdge)

        self.draw_button = tk.Button(self.left_frame, text="Draw", command=self.toggle_drawing, bg="white")
        self.draw_button.pack(pady=10)

        self.color_button = tk.Button(
            self.left_frame, text="Change Pen Color", command=self.change_color, bg="white")
        self.color_button.pack(pady=5)

        self.pen_size_frame = tk.Frame(self.left_frame, bg="gray")
        self.pen_size_frame.pack(pady=5)

        self.pen_size_1 = tk.Radiobutton(
            self.pen_size_frame, text="Small", value=3, command=lambda: self.change_size(3), bg="gray")
        self.pen_size_1.pack(side="left")

        self.pen_size_2 = tk.Radiobutton(
            self.pen_size_frame, text="Medium", value=5, command=lambda: self.change_size(5), bg="gray")
        self.pen_size_2.pack(side="left")
        self.pen_size_2.select()

        self.pen_size_3 = tk.Radiobutton(
            self.pen_size_frame, text="Large", value=7, command=lambda: self.change_size(7), bg="gray")
        self.pen_size_3.pack(side="left")
        
        self.separator = ttk.Separator(self.left_frame, orient='horizontal')
        self.separator.pack(fill='x', pady=10)

        self.text_color_frame = tk.Frame(self.left_frame, bg="gray")
        self.text_color_frame.pack(pady=5)

        self.text_color_button = tk.Button(
            self.left_frame,text="Change Text Color", command=self.change_text_color,bg="white")
        self.text_color_button.pack(padx=2)

        self.text_entry_frame = tk.Frame(self.left_frame, bg="gray")
        self.text_entry_frame.pack(pady=5)

        self.text_label = tk.Label(self.text_entry_frame, text="Enter Text:", bg="gray")
        self.text_label.pack(side="left")

        self.text_entry = tk.Entry(self.text_entry_frame)
        self.text_entry.pack(side="left", padx=5)

        self.text_size_frame = tk.Frame(self.left_frame, bg="gray")
        self.text_size_frame.pack(pady=5)

        self.text_size_label = tk.Label(self.text_size_frame, text="Text Size:", bg="gray")
        self.text_size_label.pack(side="left")

        self.text_sizes = [10, 12, 16, 20, 24, 100]  # Add more sizes if needed
        self.text_size_combobox = ttk.Combobox(self.text_size_frame, values=self.text_sizes, state="readonly")
        self.text_size_combobox.pack(side="left", padx=5)
        self.text_size_combobox.bind("<<ComboboxSelected>>", lambda event: self.change_text_size(int(self.text_size_combobox.get())))

        self.text_drag_button = tk.Button(self.left_frame, text="Enable Text Drag", command=self.toggle_text_drag, bg="white")
        self.text_drag_button.pack(pady=5)

        self.clear_button = tk.Button(self.left_frame, text="Clear",
                                command=self.clear_canvas, bg="#FF9797")
        self.clear_button.pack(pady=5)
        
        self.separator = ttk.Separator(self.left_frame, orient='horizontal')
        self.separator.pack(fill='x', pady=8)

        self.circle_button = tk.Button(self.upper_frame, text="Draw Circle", command=self.draw_circle, bg="light gray")
        self.circle_button.pack(side="left", padx=5, pady=5)

        self.rectangle_button = tk.Button(self.upper_frame, text="Draw Rectangle", command=self.draw_rectangle, bg="light gray")
        self.rectangle_button.pack(side="left", padx=5, pady=5)

        self.line_button = tk.Button(self.upper_frame, text="Draw Line", command=self.draw_line, bg="light gray")
        self.line_button.pack(side="left", padx=5, pady=5)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-3>", self.start_crop)  # Bind to right-click event
        self.canvas.bind("<B3-Motion>", self.update_crop)  # Bind to mouse motion during cropping
        self.canvas.bind("<ButtonRelease-3>", self.end_crop)

        self.rotate_button = tk.Button(self.upper_frame, text="Rotate Image", command=lambda: self.rotate_image(90), bg="light blue")
        self.rotate_button.pack(side="left", padx=5, pady=5)

        self.instruction_label = tk.Label(self.left_frame, text="Right click to crop", bg="white")
        self.instruction_label.pack(side="bottom", pady=5, padx=5, anchor="sw")

        self.scale_button = tk.Button(self.upper_frame,text="Scale Image", command=self.show_scale_dialog,bg="light blue")
        self.scale_button.pack(side="left", padx=5, pady=5)

        self.translate_button = tk.Button( self.upper_frame,text="Translate Image",command=self.show_translation_dialog,bg="light blue")
        self.translate_button.pack(side="left", padx=5, pady=5)
        
        self.Split_button = tk.Button(self.right_frame, text="Split Image", command=self.split,bg="white")
        self.Split_button.pack(pady=5)

        self.combine_button = tk.Button(self.right_frame, text="Combine Image", command=self.ask_combine_direction, bg="white")
        self.combine_button.pack(pady=5)
        
        self.separator = ttk.Separator(self.right_frame, orient='horizontal')
        self.separator.pack(fill='x', pady=10)
        
        self.split_channels_button = tk.Button(self.right_frame,text="Split Image Channels",command=self.show_split_image_dialog,bg="light gray")
        self.split_channels_button.pack(pady=5)
        
        self.bit_slicing_button = tk.Button(self.right_frame,text="Perform Bit-Slicing", command=self.bitSlicing,bg="white")
        self.bit_slicing_button.pack(pady=5)
        
        self.contour_button = tk.Button( self.right_frame, text="Find Contours", command=self.countourImage,bg="white")
        self.contour_button.pack(pady=5)
        
        self.HistogramEqualization_button = tk.Button(self.right_frame, text="Histogram Equalization", command=self.EqualizeHist,bg="white")
        self.HistogramEqualization_button.pack(pady=5)
        
        self.separator = ttk.Separator(self.right_frame, orient='horizontal')
        self.separator.pack(fill='x', pady=8)
                
        self.ImageDilation_button = tk.Button(self.right_frame, text="Image Dilation", command=self.Image_dilation,bg="white")
        self.ImageDilation_button.pack(pady=5)
        
        self.ImageErosion_button = tk.Button(self.right_frame, text="Image Erosion", command=self.Image_erosion,bg="white")
        self.ImageErosion_button.pack(pady=5)
        
        self.separator = ttk.Separator(self.right_frame, orient='horizontal')
        self.separator.pack(fill='x', pady=8)
        
        self.filter_button = tk.Button(self.right_frame,  text="Filter (Blur)", command=self.filter,bg="white")
        self.filter_button.pack(pady=5)
        
        self.sharp_button = tk.Button(self.right_frame, text="Sharpen", command=self.sharpImage,bg="white")
        self.sharp_button.pack(pady=5)
        
        self.separator = ttk.Separator(self.right_frame, orient='horizontal')
        self.separator.pack(fill='x', pady=8)
        
        self.Thresholding_button = tk.Button(self.right_frame, text="Thresholding Image", command=self.thresholdingImg,bg="white")
        self.Thresholding_button.pack(pady=5)
        
        self.AdaptiveThresholding_button = tk.Button(self.right_frame, text="Adaptive Thresholding", command=self.adaptiveThresholding,bg="white")
        self.AdaptiveThresholding_button.pack(pady=5)
        
        self.separator = ttk.Separator(self.right_frame, orient='horizontal')
        self.separator.pack(fill='x', pady=8)
        
        #self.text_label = tk.Label(self.right_frame, text="Intensity Transformation Operations", bg="gray")
        #self.text_label.pack(pady=5)
        
        self.powerLaw_button = tk.Button(self.right_frame, text="Power Law", command=self.powerLaw,bg="white")
        self.powerLaw_button.pack(pady=5)

        self.Piecewise_Linear_button = tk.Button(self.right_frame, text="Piecewise Linear", command=self.apply_contrast_stretching,bg="white")
        self.Piecewise_Linear_button.pack(pady=5)
        

    def ask_combine_direction(self):
        combine_window = tk.Toplevel(self.root)
        combine_window.title("Combine Direction")

        horizontal_button = tk.Button(combine_window, text="Horizontal", command=lambda: self.ask_number_images('horizontal'))
        horizontal_button.pack(pady=10)

        vertical_button = tk.Button(combine_window, text="Vertical", command=lambda: self.ask_number_images('vertical'))
        vertical_button.pack(pady=10)

    def ask_number_images(self, direction):
        number_window = tk.Toplevel(self.root)
        number_window.title("Number of Images to Combine")

        def combine_images(num):
            if direction == 'horizontal':
                self.combine_images_horizontal(num)
            else:
                self.combine_images_vertical(num)

        button_2 = tk.Button(number_window, text="Combine 2 Images", command=lambda: combine_images(2))
        button_2.pack(pady=10)

        button_3 = tk.Button(number_window, text="Combine 3 Images", command=lambda: combine_images(3))
        button_3.pack(pady=10)

        button_4 = tk.Button(number_window, text="Combine 4 Images", command=lambda: combine_images(4))
        button_4.pack(pady=10)
        
    def show_scale_dialog(self):
        scale_dialog = tk.Toplevel(self.root)
        scale_dialog.title("Enter Scaling Factors")

        scale_frame = tk.Frame(scale_dialog, bg="white")
        scale_frame.pack(pady=10)

        x_label = tk.Label(scale_frame, text="X Scale:", bg="gray")
        x_label.pack(side="left")

        self.x_scale_entry = tk.Entry(scale_frame)
        self.x_scale_entry.pack(side="left", padx=2)

        y_label = tk.Label(scale_frame, text="Y Scale:", bg="gray")
        y_label.pack(side="left")

        self.y_scale_entry = tk.Entry(scale_frame)
        self.y_scale_entry.pack(side="left", padx=2)

        scale_button = tk.Button(
            scale_dialog,
            text="Apply",
            command=self.scale_image,
            bg="white"
        )
        scale_button.pack(pady=5)
        
        
    def show_translation_dialog(self):
        translation_dialog = tk.Toplevel(self.root)
        translation_dialog.title("Enter Translation Offsets")

        translation_frame = tk.Frame(translation_dialog, bg="white")
        translation_frame.pack(pady=10)

        x_label = tk.Label(translation_frame, text="X Offset:", bg="gray")
        x_label.pack(side="left")

        self.x_offset_entry = tk.Entry(translation_frame)
        self.x_offset_entry.pack(side="left", padx=2)

        y_label = tk.Label(translation_frame, text="Y Offset:", bg="gray")
        y_label.pack(side="left")

        self.y_offset_entry = tk.Entry(translation_frame)
        self.y_offset_entry.pack(side="left", padx=2)

        translation_button = tk.Button(
            translation_dialog,
            text="Apply Translation",
            command=self.translate_image,
            bg="white"
        )
        translation_button.pack(pady=5)
        
        
    def show_split_image_dialog(self):
        split_dialog = tk.Toplevel(self.root)
        split_dialog.title("Choose Image Channel")

        split_frame = tk.Frame(split_dialog, bg="white")
        split_frame.pack(pady=10)

        red_button = tk.Button(split_frame, text="Red Channel", command=self.split_red_channel)
        red_button.pack(side="left", padx=5)

        green_button = tk.Button(split_frame, text="Green Channel", command=self.split_green_channel)
        green_button.pack(side="left", padx=5)

        blue_button = tk.Button(split_frame, text="Blue Channel", command=self.split_blue_channel)
        blue_button.pack(side="left", padx=5)

        combined_button = tk.Button(split_frame, text="Combined Image", command=self.combine_channels)
        combined_button.pack(side="left", padx=5)
        
        self.root.mainloop()
    
class LoadingScreen:
    
    def __init__(self):
        self.root = Tk()
        self.image = PhotoImage(file='D:/UMS/Visual Studio Code/SEM5/Assignment1/Icon.png')

        self.height = 430
        self.width = 530
        self.x = (self.root.winfo_screenwidth() // 2) - (self.width // 2)
        self.y = (self.root.winfo_screenheight() // 2) - (self.height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(self.width, self.height, self.x, self.y))
        self.root.overrideredirect(True)

        self.root.config(background="#2F6C60")

        self.welcome_label = Label(self.root, text="IMAGE EDITOR", bg="#2F6C60", font=("Trebuchet Ms", 25, "bold"),
                                   fg="#FFFFFF")
        self.welcome_label.place(x=160, y=25)

        self.bg_label = Label(self.root, image=self.image, bg="#2F6C60")
        self.bg_label.place(x=130, y=65)

        self.progress_label = Label(self.root, text="Loading...", font=("Trebuchet Ms", 13, "bold"), fg="#FFFFFF",
                                    bg="#2F6C60")
        self.progress_label.place(x=190, y=330)

        self.progress = ttk.Style()
        self.progress.theme_use('clam')
        self.progress.configure("red.Horizontal.TProgressbar", background="#108cff")

        self.progress_bar = Progressbar(self.root, orient=HORIZONTAL, length=400, mode='determinate',
                                        style="red.Horizontal.TProgressbar")
        self.progress_bar.place(x=60, y=370)
        pass
    
    def load_progress(self, i=0):
        if i <= 10:
            txt = 'Loading...' + (str(10 * i) + '%')
            self.progress_label.config(text=txt)
            self.progress_label.after(600, lambda: self.load_progress(i + 1))
            self.progress_bar['value'] = 10 * i
        else:
            self.root.destroy()
            self.open_image_editor()
            
    def open_image_editor(self): 
        self.editor = MainInterface()  
        self.root.mainloop()
        
def main():
    loading_screen = LoadingScreen()
    loading_screen.load_progress()
    loading_screen.root.mainloop()

if __name__ == "__main__":
    main()

