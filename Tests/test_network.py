import torch
from Agent.network import DQN


def verify_tensor_math():
    brain = DQN(input_channels=4, num_actions=4)
    dummy_batch = torch.zeros((32, 4, 84, 84), dtype=torch.float32)
    print("Input shape : ", dummy_batch.shape)

    try:
        q_predictions = brain(dummy_batch)
        print("Output shape : ", q_predictions.shape)
        print("Success")
    except Exception as e:
        print("Crashed : ", e)


if __name__ == "__main__":
    verify_tensor_math()
