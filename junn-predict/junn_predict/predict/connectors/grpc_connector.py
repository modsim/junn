from .model_connector import ModelConnector


class GRPCConnector(ModelConnector):
    @staticmethod
    def check_import():
        import grpc
        import tensorflow as tf
        import tensorflow_serving

    def __init__(self, arg, name=''):
        super().__init__(arg)
        # warmup
        import grpc

        # noinspection PyUnresolvedReferences
        import tensorflow as tf  # tf serving needs tf ... which is a HUGE dependency

        # noinspection PyUnresolvedReferences
        import tensorflow_serving.apis
        from tensorflow_serving.apis import prediction_service_pb2_grpc

        assert name != '', "Name must be specified"

        self.timeout = 60.0
        maximum_message_length = 100 * 1024 * 1024

        options = [
            ('grpc.max_message_length', maximum_message_length),
            ('grpc.max_receive_message_length', maximum_message_length),
        ]

        # TODO: support for secure channels?
        self.channel = grpc.insecure_channel(arg, options=options)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.name = name

    def get_signatures(self):
        from tensorflow_serving.apis import get_model_metadata_pb2

        mm = get_model_metadata_pb2.GetModelMetadataRequest()

        mm.model_spec.name = self.name
        mm.metadata_field.append('signature_def')

        result = self.stub.GetModelMetadata(mm, self.timeout)

        type_url = result.metadata['signature_def'].type_url
        value = result.metadata['signature_def'].value

        assert type_url == 'type.googleapis.com/tensorflow.serving.SignatureDefMap'

        signatures_map = get_model_metadata_pb2.SignatureDefMap()
        signatures_map.ParseFromString(value)
        signatures = list(
            sorted(
                name
                for name, signature_def in signatures_map.signature_def.items()
                if name != '__saved_model_init_op'
            )
        )

        return signatures

    def call(self, signature, data):
        import tensorflow as tf
        from tensorflow_serving.apis import predict_pb2

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = signature

        request.inputs['image'].CopyFrom(tf.make_tensor_proto(data))

        result = self.stub.Predict(request, self.timeout)
        tensor_proto = result.outputs['output_0']
        # TODO: make work for non-float32 types
        return tf.io.parse_tensor(tensor_proto.SerializeToString(), tf.float32).numpy()
