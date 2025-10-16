import tensorflow as tf
from tensorflow import keras
import h5py
import json
import numpy as np

def sanitize_layer_config(layer_config):
    """Membersihkan config layer lama agar sesuai versi TF/Keras baru"""
    if not isinstance(layer_config, dict):
        return layer_config

    # Perbaiki dtype yang lama (DTypePolicy → string biasa)
    if "dtype" in layer_config and isinstance(layer_config["dtype"], dict):
        dtype_conf = layer_config["dtype"]
        if "config" in dtype_conf and "name" in dtype_conf["config"]:
            layer_config["dtype"] = dtype_conf["config"]["name"]
        else:
            layer_config["dtype"] = "float32"

    # Perbaiki initializer lama
    for key in ["kernel_initializer", "bias_initializer"]:
        if key in layer_config and isinstance(layer_config[key], dict):
            init_conf = layer_config[key]
            if "class_name" in init_conf and "config" in init_conf:
                # Buat jadi string default saja
                layer_config[key] = init_conf["class_name"]

    # Hilangkan regularizer/constraint yang mungkin salah format
    for key in ["kernel_regularizer", "bias_regularizer",
                "activity_regularizer", "kernel_constraint", "bias_constraint"]:
        if key in layer_config and isinstance(layer_config[key], dict) and not layer_config[key]:
            layer_config[key] = None

    return layer_config

def convert_old_model(old_model_path, new_model_path):
    print(f"Converting {old_model_path}...")

    with h5py.File(old_model_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            print("Error: No model config found")
            return False

        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')

        model_config = json.loads(model_config)

        # Perbaiki batch_shape → batch_input_shape
        if 'config' in model_config and 'layers' in model_config['config']:
            for layer in model_config['config']['layers']:
                if 'config' in layer:
                    if 'batch_shape' in layer['config']:
                        batch_shape = layer['config']['batch_shape']
                        layer['config']['batch_input_shape'] = batch_shape
                        del layer['config']['batch_shape']
                        print(f"Fixed batch_shape for layer: {layer['config'].get('name', '?')}")

                    # Sanitize konfigurasi tiap layer
                    layer['config'] = sanitize_layer_config(layer['config'])

        try:
            model = keras.models.model_from_json(json.dumps(model_config))
            print("✓ Model architecture loaded")
        except Exception as e:
            print(f"✗ Error creating model: {e}")
            return False

        # Load weights jika tersedia
        if 'model_weights' in f:
            weight_layer_names = [
                n.decode('utf8') if isinstance(n, bytes) else n
                for n in f['model_weights'].attrs['layer_names']
            ]
            print(f"Loading weights for {len(weight_layer_names)} layers...")

            for layer_name in weight_layer_names:
                g = f['model_weights'][layer_name]
                weight_names = [
                    n.decode('utf8') if isinstance(n, bytes) else n
                    for n in g.attrs['weight_names']
                ]
                weights = [np.array(g[w]) for w in weight_names]

                try:
                    layer = model.get_layer(name=layer_name)
                    layer.set_weights(weights)
                    print(f"  ✓ {layer_name}")
                except Exception as e:
                    print(f"  ✗ {layer_name}: {e}")

        print(f"\nSaving converted model to {new_model_path}...")
        model.save(new_model_path)
        print("✓ Conversion complete!")

        print("\nModel Summary:")
        model.summary()

        return True

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_model.py <old_model.h5> [new_model.h5]")
        sys.exit(1)

    old_path = sys.argv[1]
    new_path = sys.argv[2] if len(sys.argv) > 2 else "converted_model.h5"

    success = convert_old_model(old_path, new_path)

    if success:
        print(f"\n✓ Model berhasil diconvert!")
        print(f"  Input: {old_path}")
        print(f"  Output: {new_path}")
    else:
        print("\n✗ Conversion gagal. Periksa error di atas.")
