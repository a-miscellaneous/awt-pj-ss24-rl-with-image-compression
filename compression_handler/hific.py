import io
import os
import urllib
from PIL import Image
import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
import gc
from .compression_handler import CompressionHandler


class TFCWrapper(CompressionHandler):
    def __init__(self, model_name, url_prefix="https://storage.googleapis.com/tensorflow_compression/metagraphs", metagraph_cache="/tmp/tfc_metagraphs"):
        super().__init__()
        self.__class__.__name__ = model_name
        self.base_name = model_name
        self.model_name = model_name
        self.url_prefix = url_prefix
        self.metagraph_cache = metagraph_cache

    def load_cached(self, filename):
        """Downloads and caches files from web storage."""
        pathname = os.path.join(self.metagraph_cache, filename)
        try:
            with tf.io.gfile.GFile(pathname, "rb") as f:
                string = f.read()
        except tf.errors.NotFoundError:
            url = f"{self.url_prefix}/{filename}"
            request = urllib.request.urlopen(url)
            try:
                string = request.read()
            finally:
                request.close()
            tf.io.gfile.makedirs(os.path.dirname(pathname))
            with tf.io.gfile.GFile(pathname, "wb") as f:
                f.write(string)
        return string

    def instantiate_model_signature(self, signature, inputs=None, outputs=None):
        """Imports a trained model and returns one of its signatures as a function."""
        string = self.load_cached(self.model_name + ".metagraph")
        metagraph = tf.compat.v1.MetaGraphDef()
        metagraph.ParseFromString(string)
        wrapped_import = tf.compat.v1.wrap_function(
            lambda: tf.compat.v1.train.import_meta_graph(metagraph), [])
        graph = wrapped_import.graph
        if inputs is None:
            inputs = metagraph.signature_def[signature].inputs
            inputs = [graph.as_graph_element(
                inputs[k].name) for k in sorted(inputs)]
        else:
            inputs = [graph.as_graph_element(t) for t in inputs]
        if outputs is None:
            outputs = metagraph.signature_def[signature].outputs
            outputs = [graph.as_graph_element(
                outputs[k].name) for k in sorted(outputs)]
        else:
            outputs = [graph.as_graph_element(t) for t in outputs]
        return wrapped_import.prune(inputs, outputs)

    def compress_image(self, input_image, rd_parameter=None):
        """Compresses an image tensor into a bitstring."""
        sender = self.instantiate_model_signature("sender")
        if len(sender.inputs) == 1:
            if rd_parameter is not None:
                raise ValueError("This model doesn't expect an RD parameter.")
            tensors = sender(input_image)
        elif len(sender.inputs) == 2:
            if rd_parameter is None:
                raise ValueError("This model expects an RD parameter.")
            rd_parameter = tf.constant(
                rd_parameter, dtype=sender.inputs[1].dtype)
            tensors = sender(input_image, rd_parameter)
            for i, t in enumerate(tensors):
                if t.dtype.is_floating and t.shape.rank == 0:
                    tensors[i] = tf.expand_dims(t, 0)
        else:
            raise RuntimeError("Unexpected model signature.")
        packed = tfc.PackedTensors()
        packed.model = self.model_name
        packed.pack(tensors)
        del sender
        del tensors
        gc.collect()
        return packed.string

    def compress(self, pil_image: Image.Image, param) -> bytes:
        self.model_name = self.base_name + f"-{param}"
        """Compresses a PIL Image and returns the compressed bytes."""
        input_image = tf.convert_to_tensor(np.array(pil_image), dtype=tf.uint8)
        input_image = tf.expand_dims(input_image, 0)
        compressed_bytes = self.compress_image(input_image)
        return compressed_bytes

    def decompress(self, compressed_bytes: bytes) -> Image.Image:
        """Decompresses bytes and returns a PIL Image."""
        packed = tfc.PackedTensors(compressed_bytes)
        receiver = self.instantiate_model_signature("receiver")
        tensors = packed.unpack([t.dtype for t in receiver.inputs])
        del packed
        gc.collect()
        for i, t in enumerate(tensors):
            if t.dtype.is_floating and t.shape == (1,):
                tensors[i] = tf.squeeze(t, 0)
        output_image, = receiver(*tensors)
        output_image = tf.squeeze(output_image, 0)
        output_image = tf.clip_by_value(output_image, 0, 255)
        output_image = tf.cast(output_image, tf.uint8).numpy()
        pil_image = Image.fromarray(output_image)
        return pil_image


if __name__ == "__main__":
    print("Testing Hific")
    model_wrapper = ModelWrapper("mbt2018-mean-mse-1")
    compressed_bytes = model_wrapper.compress(Image.open("kodim01.png"))
    decompressed_image = model_wrapper.decompress(compressed_bytes)
    print("Image decompressed")
    decompressed_image.show()
