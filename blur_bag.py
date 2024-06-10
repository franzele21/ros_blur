"""
This program is used to blur ROS bag file.
YOu will have to have file of Yolov8 weights to find the parts to blur.



Author: @franzele21
"""

import argparse

if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="Used to blur the video from a bag file")
    parser.add_argument("bag_file_path",        help="Path to the .bag file",       type=str)
    parser.add_argument("yolo_model_path",      help="Path to the Yolov8 model",    type=str)
    parser.add_argument("bag_version",          help="Version of the bagfile (either 1 or 2)",  type=int)
    parser.add_argument("-o", "--output_file",  help="Path to the output bag file", type=str)
    parser.add_argument("--frame_rate",         help="Frame sampling rate",         type=int)
    parser.add_argument("--topic",              help="Topic where the video is loated",     type=str)
    parser.add_argument("--black_box",          help="Replace the blur with a black box", action="store_true")
    parser.add_argument("--orig_mp4",           help="Make a mp4 video of the original bag video", action="store_true")
    parser.add_argument("--new_mp4",            help="Make a mp4 video of the new bag video", action="store_true")
    parser.add_argument("-v", "--verbose",      help="Program will print its progress", action="store_true")
    args = parser.parse_args()

# The rest of the imports were put here because they take too much time
# (and we don't want it to be slow just a prompting error)
from ultralytics import YOLO
import cv2
import math
from tqdm import tqdm
import numpy as np
import os
import shutil
import uuid

# modules used for opening / editing / writing ROS bag file
import rosbag
from cv_bridge import CvBridge
import bagpy as bg
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore
import sqlite3

ROS_SETUP_FILE = "/opt/ros/humble/setup.sh"

def fill_list(box_list: list, frame_rate:int=1, box_difference:int=5):
    """
    Modifies a list of lists (`box_list`) by filling in gaps and ensuring continuity between frames
    based on spatial proximity of bounding boxes.

    This function iterates through each list of boxes (representing frames) and checks for continuity 
    of each box between consecutive frames. If a box in a previous frame does not have a close match 
    in the current frame but has one in subsequent frames (within a specified `frame_rate`), a mean 
    box is created to bridge the gap.

    It will also add a box at before the first occurance of a box, for a more smooth blurring

    Parameters
    ----------
    box_list : list of list of list of float
        A list where each element is a list representing a frame of bounding boxes, and each bounding 
        box is a list of floats representing its coordinates.
    frame_rate : int, optional
        The number of frames to look ahead for matching a box from the previous frame, default is 1.
    box_difference : int, optional
        The allowed difference between the coordinates of boxes for them to be considered close, 
        default is 5.

    Returns
    -------
    list of list of list of float
        The modified `box_list` with added boxes to ensure continuity between frames.

    Examples
    --------
    >>> boxes = [
            [
                [10, 10, 20, 20],
                [20, 20, 40, 40]
            ],
            [
                [100, 100, 150, 150]
            ],
            [
                [20, 20, 40, 40]
            ],
            [
                [15, 15, 25, 25]
            ]
        ]
    >>> fill_list(boxes, frame_rate=3)
    [
        [
            [10, 10, 20, 20], 
            [20, 20, 40, 40], 
            [100, 100, 150, 150]        # was added 
        ],
        [
            [100, 100, 150, 150], 
            [12.5, 12.5, 22.5, 22.5],   # was added
            [20.0, 20.0, 40.0, 40.0]    # was added
        ],
        [
            [20, 20, 40, 40], 
            [13.75, 13.75, 23.75, 23.75]    # was added
        ],
        [
            [15, 15, 25, 25]
        ]
    ]
    """
    for i in range(1, len(box_list)-1):
        # We look at the boxes from the previous frame
        for last_boxes in box_list[i-1]:
            correspondance_now = False
            for present_boxe in box_list[i]:
                # We find a similar box in the current frame
                if np.isclose(last_boxes, present_boxe, atol=box_difference).all():
                    correspondance_now = True
            # If found, then we stop there (no need to create a box)
            if correspondance_now:
                continue
            
            # We check if the following frames resemble a box from the previous frame
            correspondance_after = False
            for j in range(frame_rate):
                if len(box_list) > i+j+1:
                    for next_boxe in box_list[i+j+1]:
                        # We find a similar box
                        if np.isclose(last_boxes, next_boxe, atol=box_difference).all():
                            correspondance_after = True
                            break
                    if correspondance_after:
                        break
            
            # If a similar box is found, then we create an approximation between
            # the box from the previous frame and the following one
            if correspondance_after:
                box_list[i].append(np.mean([last_boxes, next_boxe], axis=0).tolist())

        # We add boxes to the previous frames
        for present_boxe in box_list[i]:
            correspondance = False
            for last_boxe in box_list[i-1]:
                if np.isclose(last_boxe, present_boxe, atol=box_difference).all():
                    correspondance = True
            if not correspondance:
                for j in range(frame_rate):
                    if i-j-1 >= 0:
                        box_list[i-j-1].append(present_boxe)

    return box_list

def blur_box(frame: np.ndarray, 
             box: list, 
             black_box: bool=False):
    """
    Apply a blur or black box to a specified region of an image if the confidence level is above a threshold.

    Parameters
    ----------
    frame : numpy.ndarray
        The image on which the blur or black box is to be applied. It should be a 3D array representing an RGB image.
    box : object
        An object containing the bounding box coordinates and confidence score. It should have attributes `conf` and `xyxy`:
        - `box.conf` : list or numpy.ndarray
            The confidence score(s) of the bounding box, with values between 0 and 1.
        - `box.xyxy` : list or numpy.ndarray
            The coordinates of the bounding box in the format [x1, y1, x2, y2].
    black_box : bool, optional
        If True, the specified region is filled with a black box instead of being blurred. Default is False.
    min_conf : float, optional
        The minimum confidence threshold to apply the blur or black box. Default is 0.3. Must in [0, 1].

    Returns
    -------
    numpy.ndarray
        The modified image with the blur or black box applied to the specified region.

    Notes
    -----
    - The input `frame` must be a 3D NumPy array representing an image with shape (height, width, channels).
    - The bounding box coordinates and confidence score must be provided in the `box` object (already implemented in the Yolov8 results).
    - If `black_box` is set to True, the region within the bounding box will be replaced with black pixels (faster than blurring).
    - The Gaussian blur applied uses a kernel size of (51, 51) with a standard deviation of 0.
    """
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    h, w = y2-y1, x2-x1

    if black_box:
        blur = np.zeros((h, w, 3))
    else:
        ROI = frame[y1:y1+h, x1:x1+w]
        blur = cv2.GaussianBlur(ROI, (51,51), 0) 
    frame[y1:y1+h, x1:x1+w] = blur
    return frame

def create_video_file(output_path: str,
                      fps: float, 
                      frame_size: tuple[int, int]):
    """
    Create a video file for writing frames.

    Parameters
    ----------
    output_path : str
        The path where the output video file will be saved.
    fps : float
        The frames per second (FPS) for the output video.
    frame_size : tuple of int
        A tuple representing the width and height of the video frames in pixels.

    Returns
    -------
    video : cv2.VideoWriter
        The VideoWriter object for writing frames to the video file.

    Examples
    --------
    >>> video = create_video_file('output_video.mp4', 30.0, (1920, 1080))
    >>> for frame in frames:
    >>>     video.write(frame)
    >>> video.release()

    Notes
    -----
    - The `fps` parameter should be a positive float representing the desired frame rate for the output video.
    - The `frame_size` parameter should be a tuple of two integers representing the width and height of the video frames in pixels.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename=output_path, 
                        fourcc=fourcc, 
                        fps=fps, 
                        frameSize=frame_size)
    return video

def blur_ros1(model, 
              input_path: str, 
              output_path: str,
              img_topic: str,
              frame_verif_rate: int=5,
              black_box: bool=False,
              verbose: bool=False) -> None:
    """
    Processes an input ROS bag file to blur areas in images based on
    model predictions, and saves the processed images to a new output
    ROS bag file.

    Parameters
    ----------
    model : object
        An object representing a detection model. This model must have a
        callable method model(image, stream=True, verbose=False) that
        returns an object with a 'boxes' attribute. Each 'box' contains
        bounding box information and confidence scores.
    input_path : str
        File path to the input ROS bag file containing images to process.
    output-path : str
        File path where the output ROS bag file with blurred images will
        be saved.
    img_topic : str
        Topic name in the ROS bag file that contains the images to
        process.
    frame_verif_rate : int, optional
        Number of frames over which an intermediate verification is
        performed to decide if blurring should occur. Default is 5.
    black_box : bool, optional
        If True, uses a solid black box for blurring. Otherwise, a
        standard blurring filter is applied. Default is False.
    verbose : bool, optional
        If True, prints additional information about processing steps.
        Default is False.

    Notes
    -----
    The function iterates over messages in the input ROS bag file. When
    image messages are found under the specified `img_topic`, they are
    processed in batches defined by `frame_verif_rate` rate. For each batch,
    the central image is used for detection, and if there is any detections
    the images in the batch will be analysed by the model and possibly be 
    blurred.
    If `verbose` is True, function will output messages indicating
    processing progress and final completion.
    """
    min_conf = None             # Initialized to none but will evolve 
    bridge = CvBridge()         # Used to transform ROS images to CV2 
                                # and vice versa

    # creating the output baggile
    with rosbag.Bag(output_path, 'w') as outbag:
        last_imgs = []          # Will contain the bath of size `frame_verif_rate`
        boxes_list = []         # Will contain the boxes of the reconnized elements
        begin = True            # Used to know if it is the beginning of the process
        full_batch = False      # Will be true when the batch is full
        img_msg = None

        if verbose: print(f"\nNumber of messages: {rosbag.Bag(input_path).get_message_count()}")
        if verbose:
            input_bag_messages = tqdm(rosbag.Bag(input_path).read_messages())
        else:
            input_bag_messages = rosbag.Bag(input_path).read_messages()

        if verbose: print("\nExtracting the boxes")

        # We iterate through all messages in the input bagfile
        for topic, msg, t in input_bag_messages:
            if topic == img_topic:
                img_msg = msg       # Keep information about the image

                # Decode from a ROS image message to a CV2 image
                img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()
                last_imgs.append(img)   # Add the image to the batch

                # If it is the first frames and the batch is half full
                if begin and len(last_imgs) == math.ceil(frame_verif_rate/2):
                    begin = False
                    full_batch = True
                    idx_img = 0         # The verification frame is the first frame
                # If the batch is full (and it isn't the first frames)
                elif len(last_imgs) == frame_verif_rate:
                    full_batch = True
                    # The verifcation frame is the middle one of the batch
                    idx_img = math.ceil(frame_verif_rate/2)-1

                if full_batch:
                    full_batch = False

                    # We check if something were detected in the 
                    # verification frame 
                    boxes = next(model(last_imgs[idx_img], stream=True, verbose=False)).boxes
                    to_blur = len(boxes) > 0

                    # We iterate through the frames of the batch
                    for img in last_imgs:
                        # We first say that there is no detection in 
                        # this frame
                        boxes_list.append([])

                        # If the verification frame had a detection we 
                        # launch the detection for the frames of the batch
                        if to_blur: 
                            boxes = next(model(img, stream=True, verbose=False)).boxes

                            # If there was a detection in the frame, we 
                            # modify the thrust threeshold 
                            if len(boxes.conf) > 0:
                                if isinstance(min_conf, type(None)):
                                    min_conf = boxes.conf.mean() * 0.6
                                else:
                                    min_conf = (min_conf + boxes.conf.mean())/2 * 0.6
                            
                            # For all detections, we add those boxes 
                            # to the boxes list
                            for box in boxes:
                                if box.conf[0] > min_conf:
                                    boxes_list[-1].append(box.xyxy[0].tolist())
                    
                    # Empty the batch
                    last_imgs = []
            else:
                # We write the messages of the other topics
                outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)

        # If there is still some last frames in the batch, we make the 
        # same process as before, but the verification frame is the last frame
        if len(last_imgs) > 0:
            boxes = next(model(last_imgs[-1], stream=True, verbose=False)).boxes
            to_blur = len(boxes) > 0
            for img in last_imgs:
                boxes_list.append([])
                if to_blur:
                    boxes = next(model(img, stream=True, verbose=False)).boxes 
                    for box in boxes:
                        boxes_list[-1].append(box.xyxy[0].tolist())


        assert not isinstance(img_msg, type(None)), f"No images were found for the topic {img_topic}"

        if verbose: print("\nCountering the flickering")

        # This function adds boxes to counter flickering of the boxes, 
        # and add boxes before the prediction 
        boxes_list = fill_list(boxes_list, frame_verif_rate*2, math.ceil(math.sqrt(max(img_msg.width, img_msg.height)))*2.5)

        if verbose:
            input_bag_messages = tqdm(rosbag.Bag(input_path).read_messages())
        else:
            input_bag_messages = rosbag.Bag(input_path).read_messages()
        if verbose: print("\nWriting the images in the output bag file")

        # Here we apply the blur to the images and we write it in 
        # the output bagfile
        idx_boxes = 0               # Index of the current box
        for topic, msg, t in input_bag_messages:
            if topic == img_topic:
                img = bridge.imgmsg_to_cv2(msg).copy()      # Extract image

                # Apply all the blut boxes to the frame
                for box in boxes_list[idx_boxes]:
                    img = blur_box(img, box, black_box)
                
                # Write the frame in the bagfile
                image_message = bridge.cv2_to_imgmsg(img, encoding="rgb8")
                outbag.write(topic, image_message, msg.header.stamp)

                # Go to the next frame
                idx_boxes += 1

    if verbose: print("End of blurring")

def blur_ros2(model, 
              input_path: str, 
              output_path: str,
              img_topic: str,
              frame_verif_rate: int=5,
              black_box: bool=False,
              verbose: bool=False):
    """
    Applies blurring to specific regions in images from a ROS2 bag file based
    on detections from a model, and outputs the processed images as
    another ROS2 bag file (but as for now, just in a MP4 video file).

    Parameters
    ----------
    model : object
        An object representing a detection model. This model must be callable
        with an image parameter and return an object containing detection
        boxes with confidence scores.
    input_path : str
        The file path to the input ROS2 bag file containing the images to be
        processed.
    output_path : str
        The file path where the processed images will be saved. The result
        will be an MP4 video file.
    img_topic : str
        The topic name in the ROS2 bag file that contains the images to be
        processed.
    frame_verif_rate : int, optional
        The number of frames after which verification and potential blurring
        occur. Default is 5.
    black_box : bool, optional
        If True, blurs the regions with a solid black box; otherwise uses
        a Gaussian blur. Default is False.
    verbose : bool, optional
        If True, prints detailed logs about the processing steps. Default is
        False.

    Notes
    -----
    This function reads from the specified `img_topic` and uses the model to
    detect areas to blur. It processes frames in groups defined by
    `frame_verif_rate`. The detections determine if subsequent frames should
    be blurred. The function currently outputs the result as an MP4 video file,
    but the commented sections provide insight into how the function might be
    extended to support writing back to a ROS2 bag file.

    The function is a placeholder and indicates that further implementation is
    needed to handle various data management aspects like copying .db3 and
    .yaml files, handling non-image topics, and properly finalizing the writing
    into the ROS2 bag file.
    """
    
    print("\n\nWARNING: BE SURE TO HAVE ENOUGH DISK SPACE (df -h to see)\n\n")

    # Create temporary names for the bag file
    dst1_name = uuid.uuid4().hex + ".bag"       # Conversion of ROS2 to temp ROS1
    dst2_name = uuid.uuid4().hex + ".bag"       # Bluring of the temp ROS1 
    dst3_name = uuid.uuid4().hex                # Conversion of the temp ROS1 to temp ROS2
    dst4_name = uuid.uuid4().hex                # Copy of the original file

    # Conversion of the input ROS2 file to the ROS1 file
    if verbose: print(f"Converting ROS2 -> ROS1 on topic {img_topic}...")
    os.system(f"rosbags-convert --src {input_path} --dst {dst1_name} --include-topic {img_topic}")

    # Blurring of the new ROS1 file
    blur_ros1(model, dst1_name, dst2_name, img_topic, frame_verif_rate, black_box, verbose)
    os.remove(dst1_name)        # Delete the first ROS1 file (it isn't needed anymore)

    # Conversion of the blurred ROS1 file to a ROS2 format
    if verbose: print("Converting ROS1 -> ROS2...")
    os.system(f"rosbags-convert --src {dst2_name} --dst {dst3_name} --dst-typestore ros2_humble")
    os.remove(dst2_name)        # Delete the blurred ROS1 file (it isn't needed anymore)

    # Copy the original file
    if verbose: print(f"Copying {input_path} into {dst4_name}")
    shutil.copytree(input_path, dst4_name)

    # Delete the images from the copied file (to avoid conflicts with the newly
    # generated images)
    if verbose: print(f"Drop video column {img_topic} in {dst4_name}")
    db3_path = os.path.join(dst4_name, [x for x in os.listdir(dst4_name) if x[-4:] == ".db3"][0])
    db3_connection = sqlite3.connect(db3_path)
    db3_cursor = db3_connection.cursor()

    topic_id = db3_cursor.execute(f"SELECT id FROM topics WHERE name = '{topic}';").fetchone()[0]

    db3_cursor.execute(f"DELETE FROM messages WHERE topic_id = {topic_id};")
    db3_cursor.execute(f"DELETE FROM topics WHERE id = {topic_id};")
    db3_connection.commit()
    db3_connection.close()

    # Creation of the .yaml file used for the merge
    if verbose: print(f"Merge of {dst4_name} and {dst3_name}")
    with open("out.yaml", "w") as file:
        file.write(f"""output_bags:
- uri: {output_path}
  all_topics: true
  all_services: true
""")
    
    # Merge the two ROS2 files
    source_cmd = f"source {ROS_SETUP_FILE}"
    convert_cmd = f"ros2 bag convert -i {dst3_name} -i {dst4_name} -o out.yaml"
    os.system(f'COMMAND="{source_cmd} && {convert_cmd}"; /bin/bash -c "$COMMAND"')
    

    # Cleaning
    if verbose: print(f"Delete {dst3_name} and {dst4_name}")
    os.remove("out.yaml")
    shutil.rmtree(dst3_name)
    shutil.rmtree(dst4_name)
    
    if verbose: print("End of blurring")

def save_ros1_mp4(bagfile, topic="/camera/color/image_raw"):
    """
    Converts a specified topic in a ROS1 bag file containing image messages
    to an MP4 video file.

    Parameters
    ----------
    bagfile : str
        The file path to the ROS bag file.
    topic : str, optional
        The image topic within the ROS bag file to be converted into video.
        Default is "/camera/color/image_raw".

    Notes
    -----
    This function reads image messages from the specified topic in the ROS1
    bag file and uses OpenCV to write these images into an MP4 video file.
    The output video file will have the same name as the bag file but with
    an .mp4 extension. The frame rate of the video is determined by the
    frequency of the topic in the ROS1 bag file.

    The video is encoded using the MP4V codec, and the frame size is set
    based on the dimensions of the first image message in the topic.
    """
    bridge = CvBridge()
    bag_data = bg.bagreader(bagfile)

    images = bag_data.reader.read_messages(topic)

    fps = bag_data.topic_table[bag_data.topic_table["Topics"] == topic]["Frequency"].item()
    frame = next(images).message

    video = create_video_file(f"{bagfile}.mp4", fps, (frame.width, frame.height))

    while True:
        try:
            video.write(bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8"))
            frame = next(images).message
        except StopIteration:
            break
    video.release()


if __name__ == "__main__":
    model = YOLO(args.yolo_model_path)
    bag_file = args.bag_file_path

    output_file = "output.bag" if isinstance(args.output_file, type(None)) else args.output_file
    if ".bag" not in output_file: 
        output_file += ".bag"
    match args.bag_version:
        case 1:
            topic = "/camera/color/image_raw" if isinstance(args.topic, type(None)) else args.topic
            blur_ros1(model, bag_file, 
                    output_file, 
                    img_topic=topic,
                    frame_verif_rate=5 if isinstance(args.frame_rate, type(None)) else args.frame_rate, 
                    black_box=args.black_box, 
                    verbose=args.verbose)
            if args.new_mp4:
                if args.verbose: print(f"Creating {output_file}.mp4")
                save_ros1_mp4(output_file, topic)
                os.system(f"rmdir {output_file[:-4]}")
            if args.orig_mp4:
                if args.verbose: print(f"Creating {bag_file}.mp4")
                save_ros1_mp4(bag_file, topic)

        case 2:
            topic = "/d435i/color/image_raw" if isinstance(args.topic, type(None)) else args.topic
            blur_ros2(model, bag_file, 
                    output_file, 
                    img_topic=topic,
                    frame_verif_rate=5 if isinstance(args.frame_rate, type(None)) else args.frame_rate, 
                    black_box=args.black_box, 
                    verbose=args.verbose)
        case _:
            print("Wrong version (either 1 or 2)")


