import numpy as np
import tensorflow as tf
import traceback

MODEL = 'model_fixed.h5'

try:
    print('Loading model:', MODEL)
    model = tf.keras.models.load_model(MODEL, compile=False)
    print('Model loaded. Preparing dummy input...')
    dummy = np.random.rand(1, 224, 224, 3).astype('float32')
    preds = model.predict(dummy)
    print('Prediction shape:', getattr(preds, 'shape', None))
    import tensorflow as tf
    probs = tf.nn.softmax(preds[0]).numpy()
    top_idx = int(np.argmax(probs))
    print('Top index:', top_idx, 'Top prob:', float(probs[top_idx]))
    print('Inference OK')
    raise SystemExit(0)
except Exception as e:
    print('Inference test failed:', repr(e))
    traceback.print_exc()
    raise SystemExit(2)
