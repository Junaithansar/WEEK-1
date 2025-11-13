import os
import sys
import traceback
import h5py
import json
import tensorflow as tf

MODEL_H5 = 'model.h5'
OUT_FIXED = 'model_fixed.h5'

print('Reconstruction script starting...')
if not os.path.exists(MODEL_H5):
    print('Model file not found:', MODEL_H5)
    sys.exit(2)

try:
    # Build MobileNetV2 backbone without weights and without top
    print('Building MobileNetV2 backbone (weights=None)...')
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )

    # Build the full model matching the saved config: global avg pool + Dense(100) + Dense(15)
    inp = tf.keras.Input(shape=(224, 224, 3), name='input_1')
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d_GlobalAveragePooling2D1')(x)
    x = tf.keras.layers.Dense(100, activation='relu', name='dense_Dense1', use_bias=True)(x)
    out = tf.keras.layers.Dense(15, activation='softmax', name='dense_Dense2', use_bias=False)(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name='reconstructed_mobilenetv2')

    print('Model constructed. Attempting to load weights by name from', MODEL_H5)
    # load_weights with by_name=True will match layer names between file and model
    # This is tolerant: it will skip unmatched layers.
    model.load_weights(MODEL_H5, by_name=True)

    print('Weights loaded (by_name=True). Saving repaired model to', OUT_FIXED)
    model.save(OUT_FIXED)
    print('Saved repaired model as', OUT_FIXED)
    sys.exit(0)

except Exception as e:
    print('Reconstruction failed:', repr(e))
    traceback.print_exc()
    sys.exit(3)
