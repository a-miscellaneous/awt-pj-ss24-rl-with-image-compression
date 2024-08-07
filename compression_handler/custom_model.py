from .compression_handler import CompressionHandler
import io
from PIL import Image
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Proximal Policy Optimization


class PPO(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.policy_layers = nn.ModuleList([
            nn.Linear(128, 101),    # For first action dimension (101 values)
            # For second action dimension (65536 values)
            nn.Linear(128, 65536),
            nn.Linear(128, 101),    # For third action dimension (101 values)
            nn.Linear(128, 101),    # For fourth action dimension (101 values)
            nn.Linear(128, 5)       # For fifth action dimension (5 values)
        ])
        self.value_layer = nn.Linear(128, 1)
        self.output_dims = output_dims  # List of number of action values per dimension

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        policy_logits = [layer(x) for layer in self.policy_layers]
        value = self.value_layer(x)
        return policy_logits, value

    def get_action(self, state):
        policy_logits, _ = self.forward(state)
        actions = []
        log_probs = []
        entropies = []
        for logits in policy_logits:
            policy_dist = Categorical(logits=logits)
            action = policy_dist.sample()
            actions.append(action.item())
            log_probs.append(policy_dist.log_prob(action))
            entropies.append(policy_dist.entropy())
        return actions, torch.stack(log_probs), torch.stack(entropies)

    def evaluate_action(self, state, action):
        policy_logits, value = self.forward(state)
        log_probs = []
        entropies = []
        for i, logits in enumerate(policy_logits):
            policy_dist = Categorical(logits=logits)
            log_prob = policy_dist.log_prob(action[:, i])
            log_probs.append(log_prob)
            entropies.append(policy_dist.entropy())
        return torch.stack(log_probs, dim=1), torch.squeeze(value), torch.stack(entropies, dim=1)


class customModel(CompressionHandler):
    def __init__(self):
        super().__init__()
        self.parameter_range = range(1, 101, 2)
        self.model = self.load_model()

    def load_model(self, path="model_episode_410_reward_0.2939134241454985.pth"):
        input_dim = 15 + 1  # Example image size plus the compression ratio
        # Matching the action space dimensions
        output_dims = [101, 65536, 101, 101, 5]
        agent = PPO(input_dim, output_dims)
        agent.load_state_dict(torch.load(path))
        agent.eval()
        return agent

    def preprocess_image(self, image):
        img = image
        feature_vector = []

        # Add height and width
        h, w, c = img.shape
        feature_vector.append(h)
        feature_vector.append(w)

        # Add aspect ratio
        feature_vector.append(h / w)

        # Add mean and std of each channel
        for i in range(c):
            mean = np.mean(img[:, :, i])
            std = np.std(img[:, :, i])
            feature_vector.append(mean)
            feature_vector.append(std)

        # Add entropy
        hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-6))
        feature_vector.append(entropy)

        # Add edge density
        edges = cv2.Canny(img, 100, 200)
        edge_density = np.mean(edges)
        feature_vector.append(edge_density)

        # Add brightness
        brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        feature_vector.append(brightness)

        # Add contrast
        contrast = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        feature_vector.append(contrast)

        # Add sharpness
        sharpness = np.mean(cv2.Laplacian(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F))
        feature_vector.append(sharpness)

        # Add saturation
        saturation = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1])
        feature_vector.append(saturation)

        return feature_vector

    def compress(self, image: Image, parameter) -> bytes:
        # Convert PIL image to cv2 image
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        to_compress = cv2_image.copy()
        cv2_image = self.preprocess_image(cv2_image)
        state = np.append(cv2_image, parameter)

        # Ensure state is in the correct format
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Sample actions stochastically
        actions, _, _ = self.model.get_action(state_tensor)

        # Print actions for debugging
        print("Actions sampled:", actions)

        compression_level, rst_interval, luma_quality, chroma_quality, sampling_factor = actions
        sampling_map = {
            0: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_411,
            1: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
            2: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
            3: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_440,
            4: cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444
        }
        sampling_factor = np.clip(sampling_factor, 0, 4)
        sampling = sampling_map[int(sampling_factor)]

        # Encode the image using OpenCV with additional JPEG parameters
        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), int(compression_level),
            int(cv2.IMWRITE_JPEG_RST_INTERVAL), int(rst_interval),
            int(cv2.IMWRITE_JPEG_LUMA_QUALITY), int(luma_quality),
            int(cv2.IMWRITE_JPEG_CHROMA_QUALITY), int(chroma_quality),
            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sampling
        ]
        success, buffer = cv2.imencode('.jpg', to_compress, encode_params)
        if not success:
            raise ValueError("Failed to compress image")

        buffer = io.BytesIO(buffer)
        return buffer.getvalue()
