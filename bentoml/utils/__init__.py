# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import imp

from six.moves.urllib.parse import urlparse, uses_netloc, uses_params, uses_relative
from google.protobuf.json_format import MessageToJson, MessageToDict
from ruamel.yaml import YAML

from bentoml import __version__ as BENTOML_VERSION, _version as version_mod

try:
    from pathlib import Path

    Path().expanduser()
except (ImportError, AttributeError):
    from pathlib2 import Path

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


def is_url(url):
    try:
        return urlparse(url).scheme in _VALID_URLS
    except Exception:  # pylint:disable=broad-except
        return False


def isidentifier(s):
    """
    Return true if string is in a valid python identifier format:

    https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """
    try:
        return s.isidentifier()
    except AttributeError:
        # str#isidentifier is only available in python 3
        return re.match(r"[A-Za-z_][A-Za-z_0-9]*\Z", s) is not None


def dump_to_yaml_str(yaml_dict):
    yaml = YAML()
    string_io = StringIO()
    yaml.dump(yaml_dict, string_io)
    return string_io.getvalue()


def pb_to_yaml(message):
    message_dict = MessageToJson(message)
    return dump_to_yaml_str(message_dict)


def ProtoMessageToDict(protobuf_msg, **kwargs):
    if 'preserving_proto_field_name' not in kwargs:
        kwargs['preserving_proto_field_name'] = True

    return MessageToDict(protobuf_msg, **kwargs)


def _is_pypi_release():
    is_installed_package = hasattr(version_mod, 'version_json')
    is_tagged = not BENTOML_VERSION.startswith('0+untagged')
    is_clean = not version_mod.get_versions()['dirty']
    return is_installed_package and is_tagged and is_clean


def _is_bentoml_in_editor_mode():
    is_editor = False
    bentoml_location = Path(imp.find_module('bentoml')[1]).parent

    setup_py_path = os.path.join(bentoml_location, 'setup.py')
    if not _is_pypi_release() and os.path.isfile(setup_py_path):
        is_editor = True

    return is_editor
