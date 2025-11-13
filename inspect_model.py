import h5py, json, sys
p = 'model.h5'
print('\nInspecting', p)
try:
    with h5py.File(p, 'r') as f:
        print('\nTop-level keys:')
        print(list(f.keys()))

        print('\nTop-level attrs:')
        for k, v in f.attrs.items():
            try:
                # print small attrs nicely
                vs = v.decode() if isinstance(v, (bytes, bytearray)) else v
                print(f" - {k}: {vs}")
            except Exception:
                print(f" - {k}: (unprintable)")

        mc = f.attrs.get('model_config')
        if mc is not None:
            try:
                s = mc.decode() if isinstance(mc, (bytes, bytearray)) else str(mc)
                print('\nFound model_config attribute. Showing first 2000 chars:')
                print(s[:2000])
                # also try parse JSON
                try:
                    cfg = json.loads(s)
                    print('\nmodel_config parsed as JSON. Top-level keys:', list(cfg.keys()))
                except Exception as e:
                    print('Could not parse model_config as JSON:', e)
            except Exception as e:
                print('Could not decode model_config:', e)
        else:
            print('\nNo model_config attribute found.')
            if 'model_weights' in f:
                print("Found 'model_weights' group (weights present).")
            else:
                print("No 'model_weights' group found either.")

except Exception as e:
    print('Error opening HDF5 file:', e)
    sys.exit(2)

print('\nDone')
