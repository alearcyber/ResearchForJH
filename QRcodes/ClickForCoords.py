"""
code taken from stack overflow as an example of clicking for coordinates

DELETE THIS LINE BELOW
/Users/aidanlear/Desktop/cheese.jpg
"""

import tkinter as tk
import PIL.Image, PIL.ImageTk
import cv2


def click_for_coordinates(image):
    """
    Image can be either a PIl image or an NDarray


    Opens a window with the given image in in and allows the user to click 4 times. The function then
    returns the coordinates of the 4 points select in the form of [x1, y1, x2, y2, x3, y3, x4, y4] where
    1=topleft, 2=topright, 3=bottomright, 4=bottomleft
    """

    #create the root for tkinter
    root = tk.Tk()

    #convert to ndarray if the image is a PIl.Image
    original = image
    try:  # converts image to pil image incase it is not
        original = PIL.Image.fromarray(original)
    except:
        pass

    # setting up a tkinter canvas
    w = tk.Canvas(root, width=original.width + 100, height=original.height)
    w.pack()

    img = PIL.ImageTk.PhotoImage(original)
    w.create_image(0, 0, image=img, anchor="nw")


    # add the text
    text = tk.Text(root, height=6)
    text.insert(tk.END, 'Click to select the coordinates of the corners...\nTop Left:')
    text.pack(side=tk.RIGHT)


    next_corner_text = ['Bottom Left: ', 'Bottom Right: ', 'Top Right: ']
    coordinates = []
    def collect_coords(event):
        """
        Event for recording where on the screen the user clicks
        """
        #text.

        coordinates.append(int(event.x))
        coordinates.append(int(event.y))

        if len(coordinates) >= 8:
            w.quit()
            root.destroy()



    w.bind("<Button 1>", collect_coords)
    root.mainloop()

    #Return the coordinates the user uses.
    return coordinates



def main():
    image_path = input("Image path for image to perform homography:")   #ask user for input
    img = cv2.imread(image_path)  # read image into numpy ndarray
    coordinates = click_for_coordinates(img)  # Open the gui for the user to click the coordinates
    print(coordinates)  # see what coordinates the user clicked







    #Continue running stuff to see the crash in action
    a = int(input("side length a:"))
    b = int(input("side length b:"))
    h = int(input("hypotenuse length c:"))

    #see if it is a pythagorean triple
    print("Checking to see if it is a pythagorean triple...")
    result = a**2 + b**2 == h**2
    print("That IS a pythagorean triple" if result else "That IS NOT a pythagorean triple")

if __name__ == '__main__':
    main()
