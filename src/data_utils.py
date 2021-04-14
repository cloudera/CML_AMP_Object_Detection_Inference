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
import pickle

from src.model_utils import COCO_LABELS
from src.model_utils import get_inference_artifacts
from src.app_utils import get_feature_map_plot, get_anchor_plots, plot_predictions


def create_directory_structure(dirname):
    """ Builds skeleton folder structure to hold artifacts used in the app"""

    os.makedirs(f"data/{dirname}")
    for subdir in ["fpn", "rpn", "nms"]:
        os.makedirs(f"data/{dirname}/{subdir}")


def gather_data_artifacts(img_path):
    """
    Uses specified image path to load the image and gather all data artifacts to be used
    throughout the app.

    Args:
        img_path

    Returns:
        data_artifacts

    """

    inference_artifacts = get_inference_artifacts(img_path, False)
    feature_map_figure = get_feature_map_plot(inference_artifacts["model"])
    anchor_plots = get_anchor_plots(
        inference_artifacts["image"],
        inference_artifacts["model"].anchor_generator,
        inference_artifacts["outputs"]["boxes"],
        inference_artifacts["model"].viz_artifacts["features"],
    )
    prediction_figures = {
        k: plot_predictions(
            image=inference_artifacts["image"],
            outputs=inference_artifacts["outputs"],
            label_map=COCO_LABELS,
            nms_off=False if k == "with_nms" else True,
        )
        for k in ["with_nms", "without_nms"]
    }

    data_artifacts = {
        "outputs": inference_artifacts["outputs"],
        "image": inference_artifacts["image"],
        "feature_map_fig": feature_map_figure,
        "anchor_plots": anchor_plots,
        "prediction_figures": prediction_figures,
    }

    return data_artifacts


def create_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj