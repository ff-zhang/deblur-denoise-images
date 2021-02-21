import cv2
import numpy as np
import os

from CBDNet_Guo.SomeISP_operator_python.ISP_implement import ISP

def generate_noisy_image(path, isp):
    print("getting image")
    img = cv2.imread(path)
    np.array(img, dtype="uint8")
    img = img.astype("double") / 255.0
    img_rgb = isp.BGR2RGB(img)

    img_Irgb_gt, img_Irgb = isp.cbdnet_noise_generate_srgb(img_rgb)

    return img_Irgb_gt, img_Irgb

def dataset_add_noise(database):
    isp = ISP("CBDNet_Guo/SomeISP_operator_python/")
    path = "test\\" + database

    for root, dirs, images in os.walk(path):
        img_path = "test\\" + database + "_noisy"
        
        if not root[-3:] in os.listdir(img_path) and root[-3:].isdigit():
            os.mkdir(os.path.join(img_path, root[-3:]))

        for image in images:
            print(os.path.join(root, image))
            image_Irgb_gt, img_Irgb = generate_noisy_image(os.path.join(root, image), isp)

            print("writing image")
            img_Ibgr = isp.RGB2BGR(img_Irgb)

            print(os.path.join(img_path, root[-3:], image))
            cv2.imwrite(os.path.join(img_path, root[-3:], image), img_Ibgr*255)

            print("\n")
