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

# This module is adapted from the original Github Gist module
# found at https://gist.github.com/FranzDiebold/898396a6be785d9b5ca6f3706ef9b0bc
# It has been updated for Streamlit v1.10.0

"""Hack to add per-session state to Streamlit.

Usage
-----

>>> import SessionState
>>>
>>> session_state = SessionState.get(user_name='', favorite_color='black')
>>> session_state.user_name
''
>>> session_state.user_name = 'Mary'
>>> session_state.favorite_color
'black'

Since you set user_name above, next time your script runs this will be the
result:
>>> session_state = get(user_name='', favorite_color='black')
>>> session_state.user_name
'Mary'

"""

import os

from streamlit.scriptrunner import get_script_run_ctx
from streamlit.server.server import Server

from src.data_utils import gather_data_artifacts, create_pickle


class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    def _set_path_attributes(self):

        ROOT_PATH = f"data/{self.img_option}"

        self.img_option = self.img_option
        self.ROOT_PATH = ROOT_PATH
        self.pkl_path = f"{ROOT_PATH}/{self.img_option}.pkl"

    def _save_figure_images(self):

        fig_paths = {}

        # save feature map figure
        fpn_img_path = os.path.join(self.ROOT_PATH, "fpn", "feature_map_fig.png")
        fig_paths["fpn"] = fpn_img_path
        self.data_artifacts["feature_map_fig"].savefig(
            fpn_img_path, bbox_inches="tight"
        )

        # save anchor box figures
        rpn = {}
        for pyramid_level, data in self.data_artifacts["anchor_plots"].items():
            rpn_img_path = os.path.join(self.ROOT_PATH, "rpn", f"{pyramid_level}.png")
            rpn[pyramid_level] = rpn_img_path
            data["fig"].savefig(rpn_img_path)

        fig_paths["rpn"] = rpn

        # save final prediction figures
        nms = {}
        for nms_setting, fig in self.data_artifacts["prediction_figures"].items():
            nms_img_path = os.path.join(self.ROOT_PATH, "nms", f"{nms_setting}.png")
            nms[nms_setting] = nms_img_path
            fig.savefig(nms_img_path)

        fig_paths["nms"] = nms

        return fig_paths

    def _prepare_data_assets(self):

        self.data_artifacts = gather_data_artifacts(img_path=self.img_path)
        self.fig_paths = self._save_figure_images()
        self.has_detections = (
            True if len(self.data_artifacts["outputs"]["boxes"]) > 0 else False
        )
        self.data_artifacts.pop("outputs")


def get(**kwargs):
    """Gets a SessionState object for the current session.
    Creates a new object if necessary.
    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.
    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'
    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'
    """
    # Hack to get the session object from Streamlit.

    session_id = get_script_run_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError('Could not get Streamlit session object.')

    this_session = session_info.session

    # Got the session object! Now let's attach some state into it.

    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state

  
  