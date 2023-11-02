import time
import logging

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2


class ClarifaiPrompter:
    # based on https://github.com/isaac-chung/tweetBot98/blob/main/llm.py
    def __init__(self, user_id, app_id, pat):
        self.user_data_object = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)
        self.metadata = (('authorization', 'Key ' + pat),)

        channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(channel)

    def _predict(self, model_id, raw_text_ner):
        return self.stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=self.user_data_object,
                model_id=model_id,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            text=resources_pb2.Text(
                                raw=raw_text_ner
                            )
                        )
                    )
                ]
            ),
            metadata=self.metadata
        )

    def predict(self, model_id, raw_text_ner, retries=3) -> str:
        for i in range(retries):
            post_model_outputs_response = self._predict(model_id, raw_text_ner)
            if post_model_outputs_response.status.code == status_code_pb2.SUCCESS:
                return post_model_outputs_response.outputs[0].data.text.raw
            if i == retries - 1:
                logging.error(post_model_outputs_response.status)
                raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)
            logging.info(f"Prompt trial {i} failed. Sleeping for one minute.")
            time.sleep(60)

