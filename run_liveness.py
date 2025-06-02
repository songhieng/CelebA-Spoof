import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision

# Add the intra_dataset_code directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'CelebA-Spoof', 'intra_dataset_code'))

# Import the necessary modules
from models import AENet
from detector import CelebASpoofDetector

def pretrain(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        realname = name.replace('module.', '')
        if realname in own_state:
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            try:
                own_state[realname].copy_(param)
            except:
                print(f'While copying parameter {realname}, dimensions do not match.')

class LivenessDetector(CelebASpoofDetector):
    def __init__(self):
        self.num_class = 2
        self.net = AENet(num_classes=self.num_class)
        
        # Check if CUDA is available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the model checkpoint
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            'CelebA-Spoof',
            'intra_dataset_code',
            'ckpt_iter.pth.tar'
        )
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrain(self.net, checkpoint['state_dict'])
        
        self.new_width = self.new_height = 224
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.new_width, self.new_height)),
            torchvision.transforms.ToTensor(),
        ])
        
        self.net.to(self.device)
        self.net.eval()
    
    def preprocess_data(self, image):
        processed_data = Image.fromarray(image)
        processed_data = self.transform(processed_data)
        return processed_data
    
    def eval_image(self, image_tensors):
        # image_tensors: list of preprocessed torch.Tensors
        data = torch.stack(image_tensors, dim=0)  # shape: [N, C, H, W]
        channel = 3
        input_var = data.view(-1, channel, data.size(2), data.size(3)).to(self.device)
        with torch.no_grad():
            rst = self.net(input_var).detach()  # shape: [N, num_class]
        return rst.reshape(-1, self.num_class)
    
    def predict(self, images):
        # images: either a single ndarray or list of ndarrays (RGB)
        if not isinstance(images, list):
            images = [images]
        
        preprocessed_list = []
        for img in images:
            data = self.preprocess_data(img)
            preprocessed_list.append(data)
        
        logits = self.eval_image(preprocessed_list)                  # torch.Tensor [N, 2]
        probs = torch.nn.functional.softmax(logits, dim=1)          # [N, 2]
        return probs.cpu().numpy().copy()                           # numpy array [N, 2]

def run_liveness_detection(path):
    # Initialize the detector
    print("Initializing liveness detector...")
    detector = LivenessDetector()
    print("Detector initialized.")
    
    def process_and_print(image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prob = detector.predict(rgb)[0]  # [p_spoof, p_live]
        print(f"\nImage: {image_path}")
        print(f"  Probability of being real (live): {prob[1]:.6f}")
        print(f"  Probability of being fake (spoof): {prob[0]:.6f}")
        label = "REAL FACE (Live)" if prob[1] > 0.5 else "FAKE FACE (Spoof)"
        print(f"  Result: {label}")
    
    if os.path.isdir(path):
        # Iterate over all supported image files in the directory
        supported_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = sorted(os.listdir(path))
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in supported_ext]
        
        if not image_files:
            print(f"No supported images found in folder: {path}")
            return
        
        print(f"Found {len(image_files)} image(s) in folder: {path}")
        for fname in image_files:
            full_path = os.path.join(path, fname)
            process_and_print(full_path)
    else:
        # Assume it's a single image path
        process_and_print(path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        run_liveness_detection(input_path)
    else:
        # No argument: start webcam mode
        print("No image/folder provided. Starting webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit(1)
        
        print("Press 'q' to quit.")
        detector = LivenessDetector()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prob = detector.predict(rgb_frame)[0]  # [p_spoof, p_live]
            result = "REAL FACE (Live)" if prob[1] > 0.5 else "FAKE FACE (Spoof)"
            live_prob = prob[1]
            
            # Overlay text
            color = (0, 255, 0) if live_prob > 0.5 else (0, 0, 255)
            cv2.putText(
                frame,
                f"{result} - {live_prob:.4f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            cv2.imshow('Liveness Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
