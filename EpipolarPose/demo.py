import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

import lib.models as models
from lib.core.config import config
from lib.core.config import update_config
from lib.core.integral_loss import get_joint_location_result
from lib.utils.img_utils import convert_cvimg_to_tensor

from lib.utils.vis import drawskeleton, show3Dpose


def pose_inference_on_image(cfg_path, image):

    # Load config file
    update_config(cfg_path)
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    image_size = config.MODEL.IMAGE_SIZE[0]

    # Create Model
    model = models.pose3d_resnet.get_pose_net(config, is_train=False)
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus)
    print('Created model...')

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    print('Loaded pre-trained weights...')

    image = cv2.resize(image, (image_size, image_size))

    img_height, img_width, img_channels = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_patch = convert_cvimg_to_tensor(image)

    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])

    # apply normalization
    for n_c in range(img_channels):
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
    img_patch = torch.from_numpy(img_patch)
    preds = model(img_patch[None, ...])
    preds = get_joint_location_result(image_size, image_size, preds)[0, :, :3]

    return image, preds


def open_vid_in_camera(channel):

    cap = cv2.VideoCapture(channel)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def draw_skeleton_video(channel, cfg_path):

    cap = cv2.VideoCapture(channel)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        image, preds = pose_inference_on_image(cfg_path, frame)
        drawskeleton(image, preds, thickness=2)

        # Display the resulting frame
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def draw_skeleton_image(cfg_path, img_path):

    image = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image, preds = pose_inference_on_image(cfg_path, image)
    # fig = plt.figure(figsize=(12, 7))
    #
    # ax = fig.add_subplot('121')
    # drawskeleton(image, preds, thickness=2)
    # ax.imshow(image)
    #
    # ax = fig.add_subplot('122', projection='3d', aspect=1)
    # show3Dpose(preds, ax, radius=128)
    # ax.view_init(-75, -90)

    drawskeleton(image, preds, thickness=2)
    plt.imshow(image)
    plt.show()


def main():

    cfg_file = 'experiments/h36m/valid.yaml'
    channel = 'data/WIN_20190328_14_00_05_Pro.mp4'
    img_path = 'data/WIN_20190328_13_51_42_Pro.jpg'
    # draw_skeleton_video(channel, cfg_file)
    draw_skeleton_image(cfg_file, img_path)


if __name__ == "__main__":
    main()
