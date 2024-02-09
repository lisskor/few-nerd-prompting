import time
import logging

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

from typing import Tuple, Any

from google.protobuf.struct_pb2 import Struct


class ClarifaiPrompter:
    # based on https://github.com/isaac-chung/tweetBot98/blob/main/llm.py
    def __init__(self, user_id, app_id, pat, max_generated_tokens):
        self.user_data_object = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)
        self.metadata = (('authorization', 'Key ' + pat),)

        channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(channel)

        self.params = Struct()
        self.params.update({
            "max_tokens": max_generated_tokens
        })

    def _predict(self, model_id, raw_texts_ner):
        return self.stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=self.user_data_object,
                model_id=model_id,
                inputs=[resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(raw=t)
                    )
                ) for t in raw_texts_ner],
                model=resources_pb2.Model(
                    model_version=resources_pb2.ModelVersion(
                        output_info=resources_pb2.OutputInfo(
                            params=self.params
                        )
                    )
                )
            ),
            metadata=self.metadata
        )

    def predict(self, model_id, raw_text_ner, index, retries=3) -> Tuple[str, Tuple[Any, ...]]:
        for i in range(retries):
            post_model_outputs_response = self._predict(model_id, [raw_text_ner])
            if post_model_outputs_response.status.code == status_code_pb2.SUCCESS:
                return post_model_outputs_response.outputs[0].data.text.raw, index
            if i == retries - 1:
                logging.error(post_model_outputs_response.status)
                raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)
            logging.info(f"Prompt trial {i} failed. Sleeping for one minute.")
            time.sleep(10)
