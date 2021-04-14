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
import streamlit as st
import numpy as np

st.set_option("deprecation.showPyplotGlobalUse", False)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import SessionState
from app_pages import welcome, fpn, rpn, nms, references
from src.model_utils import COCO_LABELS
from src.app_utils import PRESET_IMAGES, APP_PAGES
from src.data_utils import load_pickle


def main():
    """
    This function acts as the scaffolding to operate the multi-page Streamlit App

    """

    st.sidebar.image("images/cloudera-fast-forward.png", use_column_width=True)
    step_option = st.sidebar.selectbox(
        label="Step through the app here:",
        options=APP_PAGES,
    )
    session_state = SessionState.get()

    if step_option == APP_PAGES[0]:

        session_state = welcome(session_state, PRESET_IMAGES)
        session_state._set_path_attributes()

    elif step_option == APP_PAGES[1]:

        with st.spinner("Hang tight while your image is processed!"):

            if (session_state.img_option not in PRESET_IMAGES.keys()) and (
                not hasattr(session_state, "data_artifacts")
            ):
                session_state._prepare_data_assets()

                if not session_state.has_detections:
                    st.error(
                        f"Sorry! The image you uploaded doesn't contain any recognizable objects. \
                        Please refresh your browser and try another image that contains one of the following classes: \
                        \n\n {', '.join([label for label in COCO_LABELS if label not in ['N/A', '__background__']])}"
                    )
            else:
                session_state = load_pickle(session_state.pkl_path)

        fpn(session_state)

    elif step_option == APP_PAGES[2]:

        if session_state.img_option in PRESET_IMAGES.keys():
            session_state = load_pickle(session_state.pkl_path)

        rpn(session_state)

    elif step_option == APP_PAGES[3]:

        if session_state.img_option in PRESET_IMAGES.keys():
            session_state = load_pickle(session_state.pkl_path)

        nms(session_state)

    elif step_option == APP_PAGES[4]:

        references()


if __name__ == "__main__":
    main()