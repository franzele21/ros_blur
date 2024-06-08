
# Automatic Video Blurring Using YOLOv8 Model

This project allows automatic blurring of a video contained in a .bag file (in ROS1 format [ROS2 not yet fully implemented]).

## Prerequisites
The project was developed using Python 3.10.12, so it is recommended to have this version or higher.

## Installation
First, clone this Git repository by running:
```
git clone https://github.com/franzele21/ros_blur.git
```
Then enter the project directory and download the necessary modules:
```
cd ros_blur
python3 -m pip install -r requirements.txt
```

## Usage
To use the program, simply type the following command in the command line:
```
python3 blur_bag.py path/to/file.bag path/to/model.pt bag_version
```

With the first argument being the path to the bag file containing the video to be blurred, the second argument is the Yolov8 model that will be used to find the areas to blur (for example, a model that can detect [faces](https://github.com/akanametov/yolov8-face) for blurring). The third argument is the bag version (either 1 or 2).

You can also add the following options:
| Command           | Alternative Form | Description                               |
|-------------------|------------------|-------------------------------------------|
| `--output_file`   | `-o`             | Path/name of the output video             |
| `--frame_rate`    |                  | Sampling interval                         |
| `--topic`         |                  | Topic where the video is saved            |
| `--black_box`     |                  | Replaces blurring with a black box (faster than blurring) |
| `--orig_mp4` |                  | Make the mp4 video of the bagfile before blurring |
| `--new_mp4` |                   | Make the mp4 video of the bagfile after blurring |
| `--verbose`       | `-v`             | Displays process progress                 |

Demonstration:

![Demo](documentation/demo.gif)

## Program Explanation

The blurring process of this programm is optimized for videos that has potentially not a lot of blurring to make, because we don't check every frame in the video (not necessary if the event isn't that frequent) but every frame in at regular intervals.


Here is a schema of how the algorithm works: 

![legend](/documentation/legende.drawio.png)

First phase, the "sampling" phase:

![phase1](/documentation/phase1_echantillonage.drawio.png)

Second phase, the "verification" phase:

![phase2](/documentation/phase2_floutage.drawio.png)

Here is a pseudocode of the blurring algorithm:

```
min_confidence = None // Minimum confidence threshold for detection

// Initialize containers for image processing
image_batch = [] // To store a batch of images
detection_boxes = [] // To store detection boxes for each image

// First phase: the "sampling" phase

// Read messages from input
FOR EACH image IN input
    APPEND image to image_batch

    // Check if a batch is ready for detection
    IF length of image_batch EQUALS frame_verif_rate THEN
		// Sampling
		SELECT middle_image from image_batch for detection


		// Perform detection using the model
		SET detection_flag to TRUE IF DETECT in middle_image
		
		// Apply detection results to all images in the batch
		// Verification phase
		FOR EACH image IN image_batch:
			IF detection_flag:
				detected_boxes = DETECT in image
				FILTER detected_boxes by confidence
				APPEND to detection_boxes
			ELSE:
				APPEND empty list to detection_boxes
			ENDIF
		ENDFOR
	ENDIF

	// Reset image batch after processing
	CLEAR image_batch
ENDFOR


// Process any remaining images in the batch
IF image_batch is not empty:
PERFORM detection on last image in batch
REPEAT steps as above for detection and storing boxes


// Apply blurring to images and write to output
FOR EACH message IN input:
    RETRIEVE corresponding detection_boxes
	IF length(detection_boxes) greater than 0 THEN
    	APPLY blurring to image using detection_boxes
	ENDIF
    WRITE image to output
ENDFOR
```

In this pseudocode, the "sampling" is done by taking the middle frame of the batch. The batch size is the given frame rate. If the sampled frame has a detection in it, then we will check for all its neighbooring frames (so it is the verification phase). 