# lint as: python3
# Copyright 2021 The Ivy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License..
# ==============================================================================
import setuptools
from pathlib import Path
from setuptools import setup
import re


def _strip(line):
    return line.split(" ")[0].split("#")[0].split(",")[0]


this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text(encoding="utf-8")
long_description = re.sub(
    r"<img [^>]*class=\"only-dark\"[^>]*>",
    "",
    long_description,
    flags=re.MULTILINE,
)

long_description = re.sub(
    r"<a [^>]*class=\"only-dark\"[^>]*>((?:(?!<\/a>).)|\s)*<\/a>\n",
    "",
    long_description,
    flags=re.MULTILINE,
)


setup(
    name="ivy-models",
    version="1.1.10",
    author="ivy",
    author_email="hello@unify.ai",
    description=(
        "Collection of pre-trained models, " "compatible with any backend framework"
    ),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://unify.ai/docs/models/",
    project_urls={
        "Docs": "https://unify.ai/docs/models/",
        "Source": "https://github.com/unifyai/models",
    },
    packages=setuptools.find_packages(),
    install_requires=[_strip(line) for line in open("requirements.txt", "r")],
    classifiers=["License :: OSI Approved :: Apache Software License"],
    license="Apache 2.0",
)
