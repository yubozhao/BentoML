from ruamel.yaml import YAML

from bentoml.cli.deployment_utils import deployment_yaml_to_pb


def test_deployment_yaml_to_pb():
    yaml = YAML()
    test_yaml = """\
  name: deployment1  
  namespace: cowork
  labels:
    labelone: value2
    labeltwo: face1t
  spec:    
    bento_name: my_model
    bento_version: 12.4k
    operator: aws_sagemaker
    sagemaker_operator_config:
      region: us-west-2
  """
    deployment_yaml = yaml.load(test_yaml)
    result_pb = deployment_yaml_to_pb(deployment_yaml)
    assert result_pb.name == 'deployment1'
