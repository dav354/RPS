import numpy as np
import time
from tflite_runtime.interpreter import Interpreter, load_delegate

def load_tpu_model(model_path="models/model_edgetpu.tflite"):
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1")]
        )
        interpreter.allocate_tensors()
        print("✅ TPU initialized successfully.")
        return interpreter, True
    except Exception as e:
        print(f"⚠️ TPU initialization failed: {e}")
        return None, False

def run_inference(interpreter, input_details, output_details, coords):
    inp = coords.flatten().astype(np.float32)
    t0 = time.time()

    if input_details[0].get('quantization', [0])[0] != 0:
        scale, zero = input_details[0]['quantization']
        inp = ((inp / scale) + zero).astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], [inp])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    infer_ms = (time.time() - t0) * 1000

    if output_details[0].get('quantization', [0])[0] != 0:
        out_scale, out_zero = output_details[0]['quantization']
        output = (output.astype(np.float32) - out_zero) * out_scale

    return output, infer_ms
