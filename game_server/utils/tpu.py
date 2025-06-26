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

# --- Start of new function for 3D alignment ---
def align_landmarks_3d(landmarks):
    """
    Aligns 3D hand landmarks:
    1. Translates the wrist (landmark 0) to the origin.
    2. Rotates the hand to a more canonical orientation.
       This example uses the vector from wrist (0) to middle finger base (9) for alignment.
       A more robust alignment might involve more points or PCA.
    """
    if landmarks.shape[1] != 3:
        raise ValueError("Landmarks must be 3D (x, y, z) for 3D alignment.")

    # Translate the wrist to the origin (landmark 0 is the wrist)
    translated_landmarks = landmarks - landmarks[0]

    # Calculate the vector from wrist (0) to middle finger base (9)
    v_wrist_mid_finger = translated_landmarks[9] - translated_landmarks[0]
    v_wrist_mid_finger_norm = np.linalg.norm(v_wrist_mid_finger)

    if v_wrist_mid_finger_norm < 1e-6: # Avoid division by zero
        return translated_landmarks

    # Normalize this vector
    v_norm = v_wrist_mid_finger / v_wrist_mid_finger_norm

    # Create target vectors (e.g., align v_norm with the Y-axis)
    target_y_axis = np.array([0, 1, 0])
    # Compute the rotation axis and angle
    rotation_axis = np.cross(v_norm, target_y_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-6: # Vectors are already aligned or opposite
        if np.dot(v_norm, target_y_axis) < 0: # Opposite, rotate 180 degrees
            rotation_angle = np.pi
            rotation_axis = np.array([1, 0, 0]) # Arbitrary axis for 180 deg
        else:
            return translated_landmarks # Already aligned
    else:
        rotation_axis = rotation_axis / rotation_axis_norm
        rotation_angle = np.arccos(np.dot(v_norm, target_y_axis))

    # Build rotation matrix using Rodrigues' rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

    # Apply rotation
    aligned_landmarks = np.dot(translated_landmarks, R.T)

    return aligned_landmarks
# --- End of new function ---

def run_inference(interpreter, input_details, output_details, coords_3d): # Renamed coords to coords_3d for clarity
    if interpreter is None:
        return None, 0.0

    # --- Start of 3D preprocessing for input consistency with training ---
    # Ensure coords_3d is already a numpy array of shape (21, 3) float32
    if coords_3d.shape != (21, 3):
        print(f"[ERROR] Expected input landmarks shape (21, 3), but got {coords_3d.shape}")
        return None, 0.0

    # 1. Apply 3D alignment (same as in training)
    coords_aligned = align_landmarks_3d(coords_3d)

    # 2. Normalize coordinates to [0, 1] range after alignment (same as in training)
    coords_min = coords_aligned.min(axis=0)
    coords_max = coords_aligned.max(axis=0)
    coords_range = coords_max - coords_min

    # Avoid division by zero for dimensions with no variation
    # Use np.where to handle cases where a dimension might have no range (e.g., all z-coords are identical)
    coords_normalized = (coords_aligned - coords_min) / np.where(coords_range > 0, coords_range, 1)

    # Flatten the 3D (x,y,z) coordinates. This will result in 21 * 3 = 63 features.
    inp = coords_normalized.flatten().astype(np.float32)
    # --- End of 3D preprocessing ---


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