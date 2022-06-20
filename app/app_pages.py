# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os
import sys
import shutil
import streamlit as st
from PIL import Image

from src.app_utils import get_feature_map_plot
from src.data_utils import create_directory_structure


def welcome(session_state, preset_images):
    """
    0. Welcome

    This function holds all streamlit content used to display the welcome page in the app.

    """

    st.title("Object Detection Inference: _Visualized_")
    st.write(
        "Object detection is a critical task in computer vision - powering use cases such as autonomous driving, surveillance, \
        defect detection in manufacturing, medical image analysis, and more. This application offers a step-by-step walkthrough to help \
        visualize the inference workflow of a single-stage object detector. Specifically, we'll see how a pre-trained [RetinaNet](https://arxiv.org/abs/1708.02002) \
        model processes an image to quickly and accurately detect objects while also exploring fundamental object detection concepts along the way."
    )

    with st.expander("Object Detection - A Brief Overview", expanded=True):
        st.write(
            "In the field of computer vision, object detection refers to the task of classifying and localizing distinct objects of interest within an image. \
            Traditionally, state-of-the-art object detectors have been based on a two-stage architecture, where the first stage narrows the search space by \
            generating a sparse set of candidate object location proposals, and the second stage then classifies the narrowed down list of proposals. While this approach yields \
            high accuracy, there is a significant tradeoff in speed, making these detectors impractical for real-time use cases."
        )
        st.write(
            "In contrast, one-stage detectors must localize and classify a much larger set of densely sampled candidate object locations all in one pass. By design, these detectors can \
            attain faster prediction speeds but must overcome the inherent challenge of disambiguating between background noise and actual \
            object signal *without* the privilege of an independent proposal system."
        )

    with st.expander("RetinaNet", expanded=True):
        st.image(
            "images/retinanet_architecture.png",
            caption="The one-stage RetinaNet network architecture",
        )
        st.write(
            "RetinaNet was the first one-stage object detection model to uphold the speed benefits of a one-stage detector while surpassing the accuracy of (at the time) all existing \
            state-of-the-art two-stage detectors. This was achieved by piecing together standard components like a Feature Pyramid Network (FPN) backbone, a Region Proposal Network (RPN), dedicated classification \
            and box regression sub-networks, and introducing a novel loss function called Focal Loss."
        )
        st.write(
            "In this application, we'll step through the inference process highlighting the inner working \
            of these fundamental concepts - select an image below to get started!"
        )

    with st.expander("Let's Get Started"):

        st.write(
            "To begin, select one of the preset images _or_ upload your own image to use throughout the application:"
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            img_setting = st.radio(
                label="Select an image setting",
                options=("Preset image", "Upload your own"),
            )

        with col2:
            if img_setting == "Preset image":
                img_option = st.selectbox(
                    "Select an preset image from the list below",
                    [key.capitalize() for key in preset_images.keys()],
                    index=0,
                )

                # display selected image
                img_option = img_option.lower()
                img_path = preset_images[img_option]
                st.image(img_path)

                session_state.img_option = img_option
                session_state.img_path = img_path

            elif img_setting == "Upload your own":
                uploaded_image = st.file_uploader(
                    label="Upload your image here",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=False,
                )

                if uploaded_image is not None:

                    # display image
                    img = Image.open(uploaded_image)
                    st.image(img, caption="Uploaded Image")

                    # create folder directory to hold assets
                    img_option = "custom"
                    parent_dir = f"data/{img_option}"

                    if os.path.exists(parent_dir):
                        shutil.rmtree(parent_dir)

                    create_directory_structure(img_option)

                    # save image to directory
                    file_type = uploaded_image.name.split(".")[-1]

                    if file_type == "png":
                        img = img.convert("RGB")

                    img_path = f"data/{img_option}/{img_option}.jpg"
                    img.save(img_path, "jpeg")

                    session_state.img_option = img_option
                    session_state.img_path = img_path

        st.info(
            "After selecting an image, use the navigation drop down menu in the top left sidebar to advance to the next page: ***1. Feature Extraction***"
        )
        st.warning(
            "**NOTE** - The pages in this application are designed for sequential use. If you return to this page after navigating away from it, you may \
            experience an FileNotFoundError, in which case, you will need to refresh the page and re-select/re-upload an image."
        )

    return session_state


def fpn(session_state):
    """
    1. Multi-scale Feature Extraction

    This function holds all streamlit content used to display the first page in the app.

    """

    st.title("1. Multi-scale Feature Extraction")
    st.write(
        "Feature extraction is central to any computer vision pipeline and is traditionally performed using deep networks of stacked convolutional layers (CNNs) \
        that refine raw images into semantically rich, low-dimensional representations. This approach _is_ transferable to object detection, however, it must \
        be improved upon to maintain scale invariant image representations because, in the real world, objects from the same class can exist at a wide range of sizes depending on their depth in an image. \
        Recognizing objects at varying scales, particularly small objects, is a fundamental challenge in object detection. \
        RetinaNet uses a [Feature Pyramid Network (FPN)](https://arxiv.org/pdf/1612.03144.pdf) to solve this problem by extracting feature maps from multiple \
        levels of a [ResNet](https://arxiv.org/pdf/1512.03385.pdf) backbone."
    )

    with st.expander("How Do FPNs Work?", expanded=False):
        st.write(
            "Feature Pyramid Networks exploit the multi-scale hierarchy that is naturally present in deep CNNs to detect objects at different scales. This is accomplished by augmenting a network's default,\
             bottom-up composition (the innate pyramidal shape) with a top-down pathway and lateral connections, as explained below."
        )
        st.image(
            "images/fpn_diagram.png",
            caption="Feature Pyramid Network Architecture",
        )

        col1, col2, col3 = st.columns(3)
        with col2:
            st.write("[Adapted Image Credit](https://arxiv.org/pdf/1612.03144.pdf)")

        st.write(
            "**a. Bottom-up Pathway:** An FPN can be constructed from any deep CNN, but RetinaNet chooses a ResNet architecture. In ResNet, convolutional layers are grouped together \
                into stages by their output size. The bottom-up pathway of the FPN simply extracts a feature map as the output from the last layer of each stage called a _pyramid level_. \
                RetinaNet constructs a pyramid with levels P$_3$ - P$_7$, where P$_l$ indicates the pyramid level and has resolution 2$^l$ lower than the input image. Each pyramid level contains 256 channels \
                and P$_0$ - P$_2$ are omitted from the FPN because their high dimensionality has a substantial impact on memory and computation speed."
        )
        st.write(
            "**b. Top-down Pathway** The top-down pathway regenerates higher resolution features by upsampling spatially coarser, but semantically stronger feature maps from higher pyramid levels. \
            Each feature map is upsampled by a factor of 2 using nearest neighbor upsampling."
        )
        st.write(
            "**c. Lateral Connections** Lateral connections between the two pathways are used to merge feature maps of the same spatial size by element-wise addition. These lateral connections combine \
                the semantically rich, upsampled feature map from the top-down pathway with accurately localized activations from the bottom-up pathway creating robust, multi-scale feature maps \
                to use for inference."
        )

    with st.expander("RetinaNet Feature Maps", expanded=True):
        st.image(session_state.img_path, caption="Original Image")
        st.subheader("Feature Maps per FPN Level")
        st.write(
            "The visual below depicts seven (of the 256 total) feature maps from each of RetinaNet's five feature pyramid levels. We see that features from the third pyramid level (P3) \
            maintain higher resolution, but semantically weaker attributes which are useful for detecting small objects. In contrast, feature maps from the final pyramid level \
            (P7) hold much lower resolution, but semantically stronger activations, making them effective for capturing larger objects."
        )
        st.image(session_state.fig_paths["fpn"], use_column_width="auto")

    return


def rpn(session_state):
    """
    2. Region Proposal Network

    This function holds all streamlit content used to display the second page in the app.

    """

    st.title("2. Inline Region Proposal Network")
    st.write(
        "Now that we've extracted features from the raw input image, the next step in the inference workflow is to generate a set of proposal \
        locations on the image that _may_ contain an object. Proposals are nothing more than rectangular-shaped candidate regions that will get \
        classified as object or not-object. But how can we capture the seemingly infinite object location, shape, and size possibilities that exist in an image?"
    )
    st.write(
        "Early approaches to object detection tackled this problem using region proposal algorithms like [selective search](https://learnopencv.com/selective-search-for-object-detection-cpp-python/) where image segmentation \
            was used to compute hierarchical groupings of similar regions based on pixel adjacency, color, texture, and size. While effective, this process \
        is computationally expensive and slow as it requires a dedicated offline algorithm to generate object proposals."
    )
    st.write(
        "To overcome these issues, the [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf) design introduced the concept of a Region Proposal Network (RPN) \
            that takes advantage of shared feature maps extracted by the CNN backbone to propose object locations in a single, end-to-end network. Like many modern \
            detectors, RetinaNet adopts the concept of _anchor boxes_ as introduced by RPN."
    )

    with st.expander("How do RPNs work?"):

        st.write("**Sliding-window Anchor Grid**")
        st.write(
            "RPN's generate proposals by overlaying a grid of anchor points onto a feature map output by the backbone network where each point corresponds to the center of one activation. \
            A set of anchor boxes is then slid over each point in the anchor grid to capture objects of variable size, height, and width. In particular, RetinaNet captures 9 anchor boxes per spatial \
            location by permuting anchor box size and aspect ratios. This means that for the 6x6 feature map in the example below, the RPN will generate 6x6x9 = 324 object proposal boxes."
        )
        st.image("images/anchor_explain.png")
        st.write("**RPN for FPN**")
        st.write(
            "As learned in the previous step, RetinaNet uses an FPN with 5 pyramid levels to extract multi-scale features. To accommodate this, the RPN independently applies a separate anchor grid to the feature maps \
            generated by each level. The anchors have sizes ranging from 32$^2$ to 512$^2$ pixels across pyramid levels P$_3$ - P$_7$ respectively. This design allows anchor boxes from lower level (higher dimensional) feature maps \
            to capture smaller objects (by pixel size), while larger anchor boxes detect objects from higher level features."
        )
        st.write("**Classification and Box Regression Subnets**")
        st.write(
            "RetinaNet attaches two independent, fully convolutional networks to the outputs of each FPN level. Each subnet shares parameters across all levels. In the classification subnet, each anchor box is responsible \
            for detecting the existence of _at most_ one object from N classes in the spatial region the anchor box covers. Similarly, in the box regression subnet, each anchor box is responsible for detecting the size and \
            shape of _at most_ one object (if any exist) by regressing the relative offset of an object's bounding box from the anchor box."
        )
        pass

    with st.expander("RPN Anchor Grid by Pyramid Level", expanded=True):

        st.write(
            "Given the explanation above, the following widget visualizes the anchor grid and anchor box sizes that are applied to feature maps at each FPN level. The top image overlays a set of 9 anchor boxes centered at _one_ anchor point for \
            each object in the image. The bottom image shows a sampled feature map from the selected FPN level to help visualize the activation granularity. By toggling the slider, we can infer which levels of the FPN are able to detect \
            each object in the image."
        )

        pyramid_level = st.select_slider(
            label="Select a Feature Pyramid Level:",
            options=[f"P{i+3}" for i in range(5)],
        )

        stats = session_state.data_artifacts["anchor_plots"][pyramid_level]["fig_stats"]

        col1, col2 = st.columns(2)
        with col1:
            st.info(
                f'**Image Size:** ({" x ".join([str(stat) for stat in stats["image_size"]])}) px \n\n **Anchor Grid:** ({" x ".join([str(stat) for stat in stats["grid_size"]])}) cells \n\n **Total # Anchors:** {stats["grid_size"][0]*stats["grid_size"][1]*9}'
            )
        with col2:
            st.info(
                f'**Anchor Sizes:** {stats["anchor_sizes"]} px$^2$ \n\n **Anchor Stride:** ({" x ".join([str(stat) for stat in stats["stride"]])}) px'
            )

        st.image(session_state.fig_paths["rpn"][pyramid_level])

    return


def nms(session_state):
    """
    3. Non-maximum Suppression

    This function holds all streamlit content used to display the final page in the app.

    """

    st.title("3. Object Detection Post Processing")

    total_anchors = sum(
        [
            grid["grid_size"][0] * grid["grid_size"][1] * 9
            for grid in [
                session_state.data_artifacts["anchor_plots"][f"P{i+3}"]["fig_stats"]
                for i in range(5)
            ]
        ]
    )

    st.write(
        f"From the previous step, we know that RetinaNet proposed and simultaneously made inference on a total of **{total_anchors:,}** anchor boxes across all pyramid levels. For this reason, it is \
        likely that objects in the image may be positively predicted by multiple anchor boxes from different pyramid levels causing duplicative detections. To refine the overlapping detections, RetinaNet \
        applies a series of post-processing steps to the model ouput."
    )

    st.write(
        "First, RetinaNet selects up to 1,000 anchor boxes from each feature pyramid level that have the highest predicted probability of any class after thresholding detector confidence at 0.05. \
        Then, the top predictions from all levels are merged together, and an algorithm called *Non-Maximum Suppression (NMS)* is applied with a threshold of 0.5 to yield the final set of non-redundant detections. \
        Finally, we can apply a confidence threshold to the remaining predictions to filter out noisy detections - here we've chosen a threshold of 0.7."
    )

    with st.expander("How does NMS work?"):
        st.write(
            "Non-Maximum Suppression takes in a refined list of predicted _bounding boxes_ across all FPN levels along with the corresponding class prediction and confidence score. Then for each class independently:"
        )

        st.write("1. Select the bounding box with the highest confidence score")
        st.write(
            "2. Compare the _Intersection over Union (IoU)_ of that bounding box with all other bounding boxes"
        )

        col1, col2, col3 = st.columns(3)

        with col2:
            with st.container():
                st.image("images/iou.png")
                st.write(
                    "      [Image Credit](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)"
                )
        st.write("3. For each box, if IoU is > 0.5, suppress that bounding box")
        st.write(
            "4. For the remaining boxes, repeat steps 1 - 3 until all bounding boxes are accounted for"
        )
        st.text("")

    with st.expander("Post-process detections", expanded=True):
        st.write(
            "The image displayed below shows the top predictions from each FPN level _before_ non-max suppression is applied. Click the \
                checkbox to see the final detections:"
        )

        cola, colb, colc = st.columns([1, 2, 1])
        with colb:
            nms_checkbox = st.checkbox("Apply Non-Max Suppression")

        if nms_checkbox:
            st.image(
                session_state.fig_paths["nms"]["with_nms"], use_column_width="auto"
            )
        else:
            st.image(session_state.fig_paths["nms"]["without_nms"])

    return


def references():
    """
    4. References

    This function holds all streamlit content used to display the references page in the app.

    """

    st.title("4. References")
    st.write(
        "In this application we’ve taken a step-by-step look at how the RetinaNet architecture pieces together common object detection components to extract robust feature maps, efficiently generate object proposal locations, and \
         post-process detections from those proposals. To dig deeper on any of the topics covered here, read more at the sources listed below - hope you’ve enjoyed it!"
    )

    st.subheader("Papers")
    st.markdown(
        """
        - [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
        - [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)
        - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
        """
    )

    st.subheader("Articles & Blogs")
    st.markdown(
        """
        - [RetinaNet Explained and Demystified](https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/#fnref4)
        - [The intuition behind RetinaNet](https://medium.com/@14prakash/the-intuition-behind-retinanet-eb636755607d#:~:text=gamma%20is%20increased.-,RetinaNet%20for%20object%20detection,%2Dthe%2Dself%20convolution%20network.)
        - [Review: FPN — Feature Pyramid Network (Object Detection)](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610)
        - [Understanding Feature Pyramid Networks for object detection (FPN)](https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)
        - [RetinaNet: how Focal Loss fixes Single-Shot Detection](https://towardsdatascience.com/retinanet-how-focal-loss-fixes-single-shot-detection-cb320e3bb0de)
        - [Faster R-CNN: Down the rabbit hole of modern object detection](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/)
        - [Region Proposal Network — A detailed view](https://towardsdatascience.com/region-proposal-network-a-detailed-view-1305c7875853)
        - [Faster R-CNN Explained for Object Detection Tasks](https://blog.paperspace.com/faster-r-cnn-explained-object-detection/)
        - [Selective Search for Object Detection](https://learnopencv.com/selective-search-for-object-detection-cpp-python/)
        """
    )