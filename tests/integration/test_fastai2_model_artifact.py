import json

import pytest
import os

import bentoml
from tests.bento_service_examples.fastai2_example_service import (
    Fastai2ExampleBentoService,
)
from tests.integration.api_server.conftest import (
    build_api_server_docker_image,
    run_api_server_docker_container,
)


test_data = {
    "age": {"0": 49},
    "workclass": {"0": " Private"},
    "fnlwgt": {"0": 101320},
    "education": {"0": " Assoc-acdm"},
    "education-num": {"0": 12.0},
    "marital-status": {"0": " Married-civ-spouse"},
    "occupation": {"0": 'engineer'},
    "relationship": {"0": " Wife"},
    "race": {"0": " White"},
    "sex": {"0": " Female"},
    "capital-gain": {"0": 0},
    "capital-loss": {"0": 1902},
    "hours-per-week": {"0": 40},
    "native-country": {"0": " United-States"},
}


@pytest.fixture(scope="module")
def fastai2_svc():
    # train
    from fastai2.data.external import untar_data, URLs
    from fastai2.tabular.core import Categorify, FillMissing, Normalize
    from fastai2.tabular.data import TabularDataLoaders
    from fastai2.tabular.learner import tabular_learner
    from fastai2.metrics import accuracy

    data_path = untar_data(URLs.ADULT_SAMPLE)
    csv_path = os.path.join(data_path, 'adult.csv')

    data_loaders = TabularDataLoaders.from_csv(
        csv_path,
        path=data_path,
        y_names="salary",
        cat_names=[
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
        ],
        cont_names=['age', 'fnlwgt', 'education-num'],
        procs=[Categorify, FillMissing, Normalize],
    )
    learn = tabular_learner(data_loaders, metrics=accuracy)
    learn.fit(1)
    svc = Fastai2ExampleBentoService()
    svc.pack('learn', learn)
    return svc


@pytest.fixture(scope="module")
def fastai2_svc_saved_dir(tmp_path_factory, fastai2_svc):
    from fastcore.utils import remove_patches_path

    tmpdir = str(tmp_path_factory.mktemp('fastai2_svc'))
    with remove_patches_path():
        fastai2_svc.save_to_dir(tmpdir)
    return tmpdir


@pytest.fixture
def fastai2_svc_loaded(fastai2_svc_saved_dir):
    return bentoml.load(fastai2_svc_saved_dir)


def test_fastai2_artifact(fastai2_svc_loaded):
    import pandas as pd
    test_data_frame = pd.read_json(json.dumps(test_data))
    result = fastai2_svc_loaded.predict(test_data_frame)
    assert (
        round(result[0][0], 2) >= 0.35 or round(result[0][0], 2) <= 0.46
    ), 'Prediction on the saved Fastai2 artifact does not match expected result'


# @pytest.fixture()
# def fastai2_image(fastai2_svc_saved_dir):
#     with build_api_server_docker_image(
#         fastai2_svc_saved_dir, 'fastai2_example_service'
#     ) as image:
#         yield image
#
#
# @pytest.fixture()
# def fastai2_docker_host(fastai2_image):
#     with run_api_server_docker_container(fastai2_image, timeout=10) as host:
#         yield host
#
#
# def test_fastai2_artifact_with_docker(fastai2_docker_host):
#     import requests
#
#     result = requests.post(
#         f"http://{fastai2_docker_host}/predict",
#         json=test_data,
#         headers={'content-Type': 'applicaiton/json'},
#     )
#     assert result.status_code == 200, 'Failed to make successful request.'
#     result = result.json()
#     print(result)
#     assert False
#     # assert (
#     #     result.json()
#     # ), 'Prediction on the Fastai2 API server does not match expected result'
