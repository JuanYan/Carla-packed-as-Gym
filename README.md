# Gym environment for Carla

## How to use

1. Install the Carla library into Python.

    CD to Carla/PythonClient folder, then type:

    ```
    pip install -e .
    ```

2. install the package, CD to gym-carla folder, then either:

    to copy & install

    ```
    pip install .
    ```

    to be able to edit the code and install:

    ```
    pip install -e . 
    ```
    
    Carla_0.8.4 fix the collision counting problem and is requred here. 

3. To use, import the gym_carla package in the code, together with gym. 

    
    ```
    import gym
    ```
    
    ```
    import gym_carla
    ```
    
    ```
    client = gym.make('Carla-v0')
    ```
    
    then it can be used by calling functions of seed, reset, step, close, render
    

4. To launch Carla, add CarlaUE4 to PATH, then use the script launch_carla.bat or launch_carla.sh to launch Carla.
