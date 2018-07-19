import gym
import gym_carla
import numpy as np
if __name__ == "__main__":
    client = gym.make('Carla-v0')
    client.seed(0)
    image = client.reset()
    action = np.array([0, 0, 0])
    for i in range(100):
        image, reward, done, measurements = client.step(action)
        client.render()
        action = measurements['autopilot_control']
        print(f'step: {i}, steer: {action.steer:.3}, throttle: {action.throttle}, brake:{action.brake}')
        if done:
            print("Done")
            break

    client.close()