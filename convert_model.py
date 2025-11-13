import h5py, json, sys, os
import tensorflow as tf

p = 'model.h5'
print('Attempting to repair model at', p)
if not os.path.exists(p):
    print('Model file not found:', p);
    sys.exit(2)

with h5py.File(p, 'r') as f:
    mc = f.attrs.get('model_config')
    if mc is None:
        print('No model_config found in HDF5. Cannot proceed.')
        sys.exit(2)
    s = mc.decode() if isinstance(mc, (bytes, bytearray)) else str(mc)
    cfg = json.loads(s)

print('Parsed model_config. Cleaning DepthwiseConv2D "groups" entries...')

def clean_groups(obj):
    if isinstance(obj, dict):
        # If this dict is a layer config with a class_name
        # remove any 'groups' entries from layer configs (DepthwiseConv2D/Conv2D etc.)
        if 'config' in obj and isinstance(obj['config'], dict):
            if 'groups' in obj['config']:
                print('Removing groups from layer:', obj.get('config', {}).get('name'))
                obj['config'].pop('groups', None)
        # recurse into values
        for k, v in obj.items():
            clean_groups(v)
    elif isinstance(obj, list):
        for item in obj:
            clean_groups(item)

clean_groups(cfg)

# Attempt to rebuild model from config
print('Attempting to rebuild model from modified config...')
try:
    # Import the proper builder to avoid conflicts with a standalone 'keras' package
    from tensorflow.keras.models import model_from_json
    model_json = json.dumps(cfg)
    # Provide a small custom_objects mapping so nested 'Sequential' / 'Functional'
    # entries in the saved config can be resolved to TF Keras classes.
    custom_objects = {
        'Sequential': tf.keras.models.Sequential,
        'Functional': tf.keras.models.Model,
        'InputLayer': tf.keras.layers.InputLayer,
    }
    model = model_from_json(model_json, custom_objects=custom_objects)
    print('Model object constructed. Now loading weights from HDF5...')
    # load weights from the original HDF5
    model.load_weights(p)
    out = 'model_fixed.h5'
    model.save(out)
    print('Successfully saved repaired model to', out)
    sys.exit(0)
except Exception as e:
    import traceback
    print('Failed to rebuild/load weights:', repr(e))
    traceback.print_exc()
    print('\nFalling back to direct load with compatibility wrappers...')
    try:
        # Define simple compatibility wrappers that accept and ignore 'groups'
        from tensorflow.keras.layers import Conv2D as KConv2D, DepthwiseConv2D as KDepthwiseConv2D

        class Conv2DCompat(KConv2D):
            def __init__(self, *args, groups=None, **kwargs):
                # ignore groups (most older models used groups=1)
                super().__init__(*args, **kwargs)

        class DepthwiseConv2DCompat(KDepthwiseConv2D):
            def __init__(self, *args, groups=None, **kwargs):
                super().__init__(*args, **kwargs)

        custom = {
            'Conv2D': Conv2DCompat,
            'DepthwiseConv2D': DepthwiseConv2DCompat,
        }
        print('Attempting tf.keras.models.load_model with custom objects...')
        from tensorflow.keras.models import load_model
        m = load_model(p, custom_objects=custom, compile=False)
        out = 'model_fixed.h5'
        m.save(out)
        print('Successfully saved repaired model to', out)
        sys.exit(0)
    except Exception as e2:
        print('Fallback load also failed:', repr(e2))
        import traceback as _tb
        _tb.print_exc()
        sys.exit(4)
