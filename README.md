# Gym environment for Carla

## How to use

1. install the package, cd to gym-carla folder, then either:

    to copy & install

    ```bash
    pip install .
    ```

    to be able to edit the code and install:

    ```bash
    pip install -e . 
    ```

2. To use, import the gym_carla package in the code, together with gym. 

    
    ```python
    import gym
    ```
    
    ```python
    import gym_carla
    ```
    
    ```python
    client = gym.make('Carla-v0')
    ```
    
    then it can be used by calling functions of seed, reset, step, close, render
    

3. To launch Carla, add CarlaUE4 to PATH, then use the script launch_carla.bat or launch_carla.sh to launch Carla.
