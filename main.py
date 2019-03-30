import os
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt


class CSVLine:
    image_name = ''
    center_x_1 = ''
    center_y_1 = ''
    polomer_1 = ''
    center_x_2 = ''
    center_y_2 = ''
    polomer_2 = ''
    center_x_3 = ''
    center_y_3 = ''
    polomer_3 = ''
    center_x_4 = ''
    center_y_4 = ''
    polomer_4 = ''
    image = None

    def __init__(self):
        pass

    def constructor(self, image_name, center_x_1, center_y_1, polomer_1, center_x_2, center_y_2, polomer_2, center_x_3,
                    center_y_3, polomer_3, center_x_4, center_y_4, polomer_4, image):
        self.image_name = image_name
        self.center_x_1 = int(center_x_1)
        self.center_y_1 = int(center_y_1)
        self.polomer_1 = int(polomer_1)

        self.center_x_2 = int(center_x_2)
        self.center_y_2 = int(center_y_2)
        self.polomer_2 = int(polomer_2)

        self.center_x_3 = int(center_x_3)
        self.center_y_3 = int(center_y_3)
        self.polomer_3 = int(polomer_3)

        self.center_x_4 = int(center_x_4)
        self.center_y_4 = int(center_y_4)
        self.polomer_4 = int(polomer_4)

        self.image = image


def load_image(path):
    img = cv2.imread(path)
    if img is not None:
        return img


def load_csv():
    csv_file = {}

    path = 'iris_NEW/iris_bounding_circles.csv'
    main_directory = "iris_NEW"

    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        for i, line in enumerate(reader):
            if i > 0:
                line_csv = CSVLine()
                line_tmp = line[0].replace('_n2', '')
                image = load_image(main_directory + '/' + line[0])

                line_csv.constructor(line_tmp, line[1], line[2], line[3], line[4], line[5], line[6],
                                     line[7],
                                     line[8], line[9], line[10], line[11], line[12], image)
                csv_file[line_tmp] = line_csv
    return csv_file

def check_if_is_in_lids(x, y, centers, range):
    return (x - centers[0]) ** 2 + (y - centers[1]) ** 2 < range ** 2


def get_circle_length(radius):
    return 2 * np.pi * radius

def process_image(image_csv):
    samples = np.linspace(0, 2 * np.pi, num=360)

    main_directory = "iris_NEW"
    image_normal = cv2.imread(main_directory + '/' + image_csv.image_name)

    center_zr = (image_csv.center_x_1, image_csv.center_y_1)
    center_du = (image_csv.center_x_2, image_csv.center_y_2)
    center_hv = (image_csv.center_x_3, image_csv.center_y_3)
    center_dv = (image_csv.center_x_4, image_csv.center_y_4)

    centerx = np.linspace(center_zr[0], center_du[0], image_csv.polomer_2 - image_csv.polomer_1)
    centery = np.linspace(center_zr[1], center_du[1], image_csv.polomer_2 - image_csv.polomer_1)

    norm = np.zeros((image_csv.polomer_2 - image_csv.polomer_1, 360))
    black_a_whit = np.zeros((image_csv.polomer_2 - image_csv.polomer_1, 360))
    polar = np.zeros((image_csv.polomer_2 - image_csv.polomer_1, 360))

    for r in range(image_csv.polomer_2 - image_csv.polomer_1):
        try:
            for it, theta in enumerate(samples):

                x = int((r + image_csv.polomer_1) * np.cos(theta) + centerx[r])
                y = int((r + image_csv.polomer_1) * np.sin(theta) + centery[r])

                norm[r][it] = image_normal[y][x][0]
                if check_if_is_in_lids(x, y, center_hv, image_csv.polomer_3) and check_if_is_in_lids(x, y, center_dv,
                                                                                                     image_csv.polomer_4):
                    black_a_whit[r][it] = 0
                    polar[r][it] = image_normal[y][x][0]
                else:
                    black_a_whit[r][it] = 255
                    polar[r][it] = 255

        except IndexError as e:
            pass
    return norm
    # plt.imshow(norm, cmap='gray')
    # plt.show()
    # plt.imshow(black_a_whit, cmap='gray')
    # plt.title("maska")
    # plt.show()
    # plt.imshow(polar, cmap='gray')
    # plt.title("odstranenie viecok")
    # plt.show()


def build_filters():
    filters = []
    ksize = 40
    for theta in np.arange(0, np.pi, np.pi / 16):
        # kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kernel = cv2.getGaborKernel((ksize, ksize), 4, theta, 8, 1, 0, ktype=cv2.CV_32F)
        kernel /= 1.5 * kernel.sum()
        filters.append(kernel)
    return filters


def process_image_with_gabor(image, filters):
    accum = np.zeros_like(image)
    equalize_hist = cv2.equalizeHist(np.uint8(image))
    for kern in filters:
        filtered_image = cv2.filter2D(equalize_hist, cv2.CV_8UC3, kern)
        accum = +filtered_image/len(filtered_image)
        # np.average(accum, filtered_image, accum)
    return accum


# loading images
print('Loading')
loaded_images = load_csv()
print('Building Filters')
filters = build_filters()
print('Processing')
for key, value in loaded_images.items():
    process_image(value)
    tmp_result = process_image(value)
    result = process_image_with_gabor(tmp_result, build_filters())
    plt.imshow(result, cmap='gray')
    plt.title(value.image_name)
    plt.show()
    plt.imshow(value.image)
    plt.title('Klasik')
    plt.show()

# for image in images:
#     result = process_image(image , build_filters())
#     cv2.imshow(image.image_name , result)
