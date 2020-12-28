from commonfunctions import *
import cv2
import os
import sys

def integral_image(img):
    rows, cols = img.shape
    int_img = np.zeros(img.shape)
    int_img[0][0] = img[0][0]
    for r in range(1, rows):
        int_img[r][0] = int_img[r-1][0] + img[r][0]
    for c in range(1, cols):
        int_img[0][c] = int_img[0][c-1] + img[0][c]
    for r in range(1, rows):
        for c in range(1, cols):
            int_img[r][c] = img[r][c]+int_img[r][c-1] + \
                int_img[r-1][c] - int_img[r-1][c-1]
    return int_img


def Segment(img, window, t=10):
    rows, cols = img.shape
    output = np.zeros(img.shape, dtype="uint8")
    int_img = integral_image(img)
    s = int(window/2)
    p_img = np.pad(img, s, "constant")
    p_int = np.pad(int_img, s, 'edge')
    for r in range(s+1, rows+s):
        for c in range(s+1, cols+s):
            x1 = c-s
            x2 = c+s
            y1 = r-s
            y2 = r+s
            count = (x2-x1)*(y2-y1)
            sum = p_int[y2, x2] - p_int[y2, x1-1] - \
                p_int[y1-1, x2] + p_int[y1-1, x1-1]
            if(img[r-s][c-s]*count) <= (sum*(100-t)/100):
                output[r-s][c-s] = 0
            else:
                output[r-s][c-s] = 255
    return output


def load_images_from_folder(folder):
    images = []
    files = []
    print(folder)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            files.append(filename)
            images.append(img)
    return images , files

if len(sys.argv) != 3:
    print('Wrong input.\n')
    exit(1)

input_folder = sys.argv[1]
output_folder = sys.argv[2]
images , files = load_images_from_folder(input_folder)
for i,img in enumerate(images):
    output = Segment(rgb2gray(img),7,35)
    io.imsave(output_folder+'/'+str(files[i]),output)
    print(i)

print(input_folder, output_folder)
