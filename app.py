from flask import Flask, request, render_template
from PIL import Image
import io
import torch
from torchvision import transforms, models
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load your trained model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # adjust num classes
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Grad-CAM class definition here (copy from your existing code)...
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.cpu().numpy()

        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

target_layer = model.layer4[-1].conv3
grad_cam = GradCAM(model, target_layer)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0)

        # Predict
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
        label_map = {0: "Normal", 1: "Pneumonia"}
        prediction = label_map[pred]

        # Generate Grad-CAM heatmap
        cam = grad_cam.generate(input_tensor, class_idx=pred)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, img.size)

        img_np = np.array(img)
        overlay = heatmap * 0.4 + img_np * 0.6
        overlay = overlay.astype(np.uint8)

        # Convert overlay to base64 string
        overlay_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format='JPEG')
        buf.seek(0)
        img_bytes = buf.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        return render_template('index.html', prediction=prediction, heatmap_img=img_b64)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
