import sys
import traceback
import tensorflow as tf

MODEL = 'model_fixed.h5'
try:
    print('Loading', MODEL)
    m = tf.keras.models.load_model(MODEL, compile=False)
    print('\nModel summary:')
    m.summary()
    print('\nOK: loaded', MODEL)
    sys.exit(0)
except Exception as e:
    print('FAILED to load model:', repr(e))
    traceback.print_exc()
    sys.exit(2)
