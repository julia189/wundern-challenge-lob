import os
import sys
import numpy as np

# Adjust path to import utils from parent directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint, ScorerStepByStep
import onnxruntime as ort

class PredictionModel:
    """
    
    This solution uses an LSTM trained on 32 features
    to predict t0 and t1.

    """

    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.sequence_history = []
        
        # Determine paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, "2026-02-25_lstm_v1.onnx")
        
        # Initialize ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.ort_session = None
        self.input_name = None

        try:
            self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
            self.input_name = self.ort_session.get_inputs()[0].name
            self.hidden_name = self.ort_session.get_inputs()[1].name
            self.memory_name = self.ort_session.get_inputs()[2].name
            
            self.output_name = self.ort_session.get_outputs()[0].name
            self.hidden_output_name = self.ort_session.get_outputs()[1].name
            self.memory_output_name = self.ort_session.get_outputs()[2].name
            
            print(f"Loaded ONNX model from {onnx_path}")
                
        except Exception as e:
            print(f"Error loading model resources: {e}")
            self.ort_session = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Update history
        self.sequence_history.append(data_point.state.copy())

        # If prediction not needed yet, return None
        if not data_point.need_prediction:
            return None
            
        if self.ort_session is None:
            return np.zeros(2)

        # Prepare input window (last 100 steps)
        # The model was trained with a context window of 100
        history_window = self.sequence_history[-100:]
        
        # Pad with zeros if history is shorter than 100 (should not happen for need_prediction=True starting at step 99)
        if len(history_window) < 100:
             padding = [np.zeros_like(history_window[0])] * (100 - len(history_window))
             history_window = padding + history_window

        data_arr = np.asarray(history_window, dtype=np.float32)
        
        # Add batch dimension: (1, Sequence_Length, Features)
        data_tensor = np.expand_dims(data_arr, axis=0)

        
        #device = torch.device(0 if torch.cuda.is_available() else 'cpu')
        batch_size = data_tensor.shape[0]
        hidden = np.zeros((1, batch_size, 128), dtype=np.float32)
        memory = np.zeros((1, batch_size, 128), dtype=np.float32)

         # Run inference
        ort_inputs = {self.input_name: data_tensor.astype(np.float32), self.hidden_name: hidden, self.memory_name: memory}
        
        output = self.ort_session.run([self.output_name, 
                                       self.hidden_output_name, 
                                       self.memory_output_name], ort_inputs)[0]
        
        if len(output.shape) == 3:
            # If model returns (Batch, Seq, Features)
            prediction = output[0, -1, :]
        else:
            # If model returns (Batch, Features)
            prediction = output[0]
            
        return prediction


if __name__ == "__main__":
    # Local testing
    test_file = f"{CURRENT_DIR}/datasets/valid.parquet"
    
    if os.path.exists(test_file):
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        print("Testing LSTM Model")
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for i, target in enumerate(scorer.targets):
            print(f"  {target}: {results[target]:.6f}")
    else:
        print("Valid parquet not found for testing.")
