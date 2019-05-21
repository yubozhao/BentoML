import os
import sys
import pytest

from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bentoml.utils.exceptions import BentoMLException
from bentoml.deployment.serverless import check_serverless_compatiable_version, \
  ServerlessDeployment  # noqa: E402


def test_serverless_version():
    with patch('bentoml.utils.whichcraft.which', return_vaue='Not None'):
        with patch(
            'bentoml.deployment.serverless.invoke_serverless_command',
            return_value=['1.40.0']
        ):
            result = check_serverless_compatiable_version()
            assert result == True

        with patch(
            'bentoml.deployment.serverless.invoke_serverless_command',
            return_value=['1.20.0']
        ):
            with pytest.raises(ValueError):
                check_serverless_compatiable_version()


def test_deploy_serverless(bento_archive_path):
    with patch('bentoml.utils.whichcraft.which', return_vaue='Not None'):
        with patch.object(ServerlessDeployment, '_generate_bundle', return_value='fake_output'):
            deployment = ServerlessDeployment(bento_archive_path, 'aws-lambda', 'us-west-2', 'dev')
            with patch(
                'bentoml.deployment.serverless.invoke_serverless_command',
                return_value=['Service Information', 'result']
            ):
                result = deployment.deploy()
                assert result == 'fake_output'

            with patch(
                'bentoml.deployment.serverless.invoke_serverless_command',
                return_value=['Incorrect response', 'result']
            ):
                with pytest.raises(BentoMLException):
                    deployment.deploy()


def test_delete_serverless(bento_archive_path):
    with patch('bentoml.utils.whichcraft.which', return_vaue='Not None'):
        deployment = ServerlessDeployment(bento_archive_path, 'aws-lambda', 'us-west-2', 'dev')

        with patch(
            'bentoml.deployment.serverless.invoke_serverless_command',
            return_value=['Serverless: Incorrect message']
        ):
          result = deployment.delete()
          assert result == False

        with patch(
            'bentoml.deployment.serverless.invoke_serverless_command',
            return_value=['Serverless: Stack removal finished...']
        ):
          result = deployment.delete()
          assert result == True