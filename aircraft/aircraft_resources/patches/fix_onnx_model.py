import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Fix ONNX model shapes.")
    parser.add_argument("input_model", type=str, help="Path to the input ONNX model")
    args = parser.parse_args()

    model_path = args.input_model

    if not os.path.exists(model_path):
        print(f"Error: Input file '{model_path}' not found.")
        sys.exit(1)

    try:
        model = onnx.load(model_path)
        new_model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        onnx.save(new_model, model_path)
        print(f"Success! Saved fixed model")

    except Exception as e:
        print(f"Failed to process model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
