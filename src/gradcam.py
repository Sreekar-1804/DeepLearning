
import numpy as np
import cv2
import torch
import torch.nn.functional as F


IMAGE_SIZE = 224


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(self.save_activations)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, task="gender", target_class=None):
        self.model.eval()
        self.model.zero_grad()

        gender_output, age_output = self.model(input_tensor)

        if task == "gender":
            output = gender_output
        elif task == "age":
            output = age_output
        else:
            raise ValueError("task must be either 'gender' or 'age'")

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        target_score = output[:, target_class]
        target_score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min != 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.4):
    """
    Overlays Grad-CAM heatmap on the original image.
    """
    original_image = original_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    original_np = np.array(original_image)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


def generate_gradcam(image, model, transform, device):
    """
    Generates gender and age Grad-CAM overlays for one PIL image.
    """
    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    target_layer = model.backbone.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    gender_cam, gender_target_class = gradcam.generate_cam(
        input_tensor=input_tensor,
        task="gender"
    )

    age_cam, age_target_class = gradcam.generate_cam(
        input_tensor=input_tensor,
        task="age"
    )

    gradcam.remove_hooks()

    gender_overlay = overlay_heatmap_on_image(image, gender_cam)
    age_overlay = overlay_heatmap_on_image(image, age_cam)

    return gender_overlay, age_overlay
