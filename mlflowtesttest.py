import mlflow
from mlflow.tracking import MlflowClient
import common as com

param = com.yaml_load('/home/rnd/Anomaly_detection/config.yaml')

class RemoteRegistry:
    def __init__(self, tracking_uri=None, registry_uri=None):
        self.server = MlflowClient(tracking_uri, registry_uri)

    def create_registered_model(self, name):
        registered_model = self.server.create_registered_model(name)
        return registered_model

    def update_registered_model(self, name, new_name=None, description=None):
        self.server.update_registered_model(name, new_name, description)

    def get_registered_model(self, name):
        registered_model = self.server.get_registered_model(name)
        return registered_model

    def get_model_version_download_uri(self, name, version):
        download_uri = self.server.get_model_version_download_uri(name, version)
        return download_uri

    def download_artifact(self, run_id, path, dst_path=None):
        return self.server.download_artifacts(run_id, path, dst_path)

    def get_last_model(self, registered_model, stage='Production'):
        for model in registered_model.latest_versions:
            if model.current_stage == stage:
                return model

class SegmentationModel():
    def __init__(self, tracking_uri, model_name):

        self.registry = RemoteRegistry(tracking_uri=tracking_uri)
        self.model_name = model_name
        self.model = self.build_model(model_name)

    def get_latest_model(self, model_name):
        registered_models = self.registry.get_registered_model(model_name)
        last_model = self.registry.get_last_model(registered_models)
        local_path = self.registry.download_artifact(last_model.run_id, 'model', './')
        return local_path

    def build_model(self, model_name):
        local_path = self.get_latest_model(model_name)

        return mlflow.keras.load_model(local_path)

    def predict(self, file_path):
        data = self.preprocess(file_path)
        result = self.model.predict(data)
        return self.postprocess(result)

    def preprocess(self, file_path):
        data = com.file_to_vector_array(file_path,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])
        return data

    def postprocess(self, result):
        return result