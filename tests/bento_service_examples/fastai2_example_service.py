import bentoml
from bentoml.adapters import DataframeInput
from bentoml.artifact import Fastai2ModelArtifact


# @bentoml.env(auto_pip_dependencies=True)
@bentoml.env(pip_dependencies=['fastai2', 'IPython', 'fastcore', 'pandas'])
@bentoml.artifacts([Fastai2ModelArtifact('learn')])
class Fastai2ExampleBentoService(bentoml.BentoService):
    @bentoml.api(input=DataframeInput())
    def predict(self, df):
        data_loaders = self.artifacts.learn.dls.test_dl(df)
        result = self.artifacts.learn.get_preds(dl=data_loaders)
        result_list = result[0].tolist()
        print(result_list)
        return result_list

