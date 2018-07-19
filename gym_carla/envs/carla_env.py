import logging
import gym
from gym_carla.envs import carla_config
import numpy as np
from carla.client import CarlaClient
from carla.sensor import Camera
from carla.settings import CarlaSettings
from gym import spaces
from gym.envs.classic_control.rendering import SimpleImageViewer
import pyglet

logger = logging.getLogger(__name__)


# from gym.envs.classic_control.rendering
class ImageViewer(SimpleImageViewer):
    def __init__(self, display=None):
        super(ImageViewer, self).__init__(display)

    def imshow(self, arr):
        if self.window is None:
            height, width, _channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display,
                                               vsync=False, resizable=True)
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0, width=self.window.width, height=self.window.height)
        self.window.flip()


class CarlaEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    reward_range = (-np.inf, np.inf)
    spec = None
    action_space = None
    observation_space = None

    def __init__(self, ):
        # action space range: steer, throttle, brake
        min_steer, max_steer = -1, 1
        min_throttle, max_throttle = 0, 1
        min_brake, max_brake = 0, 1
        self.action_space = spaces.Box(low=np.array([min_steer, min_throttle, min_brake]),
                                       high=np.array([max_steer, max_throttle, max_brake]),
                                       dtype=np.float32)
        # observation, 3 channel image
        self.observation_space = spaces.Box(low=0, high=1.0,
                                            shape=(3, carla_config.CARLA_IMG_HEIGHT, carla_config.CARLA_IMG_WIDTH),
                                            dtype=np.float32)

        self.viewer = ImageViewer()
        self.rng = np.random.RandomState()  # used for random number generators

        self.target = np.array(carla_config.TARGET)
        self.cur_image = None
        self.cur_measurements = None

        self.carla_client = CarlaClient(carla_config.CARLA_HOST_ADDRESS, carla_config.CARLA_HOST_PORT, timeout=100)
        self.carla_client.connect()
        if self.carla_client.connected():
            print("Successfully connected to", end=" ")
        else:
            print("Failed to connect", end=" ")
        print(f"Carla on {carla_config.CARLA_HOST_ADDRESS}:{carla_config.CARLA_HOST_PORT}")

    def render(self, mode='human'):
        """Renders the environment.

        :param mode: - human: render to the current display or terminal and
                      return nothing. Usually for human consumption.
                    - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
                      representing RGB values for an x-by-y pixel image, suitable
                      for turning into a video.
        :return:
        """
        if mode == 'rgb_array':
            return self.cur_image
        elif mode == 'human':
            self.viewer.imshow(arr=self.cur_image.transpose([1, 2, 0]))
            return self.viewer.isopen
        else:
            super(CarlaEnv, self).render(mode=mode)  # just raise an exception

    def step(self, action):
        """
        Run one time step of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        :param action: an action provided by the environment, dict of control signals, such as {'steer':0, 'throttle':0.2}, or array
        :return: observation (object): agent's observation of the current environment
                reward (float) : amount of reward returned after previous action
                done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
                info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if isinstance(action, dict):
            self.carla_client.send_control(**action)
        elif isinstance(action, np.ndarray):  # parse control from array
            self.carla_client.send_control(steer=action[0], throttle=action[1], brake=action[2])
        else:
            self.carla_client.send_control(action)

        # read measurements and image from Carla
        measurements, image = self._read_data()

        # calculate reward
        reward = self._reward(self.cur_measurements, measurements)
        done = self._done(measurements)

        # save current measurements for next use
        self.cur_measurements = measurements
        self.cur_image = image

        return image, reward, done, measurements

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        :return: observation (object): the initial observation of the space.
        """
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=20,
            NumberOfPedestrians=40,
            WeatherId=self.rng.choice([1, 3, 7, 8, 14]),
            QualityLevel='Low')

        # CAMERA
        camera0 = Camera('CameraRGB', PostProcessing='SceneFinal')
        # Set image resolution in pixels.
        camera0.set_image_size(800, 600)
        # Set its position relative to the car in meters.
        camera0.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera0)

        scene = self.carla_client.load_settings(settings)

        self.cur_measurements = None
        self.cur_image = None

        # define a random starting point of the agent for the appropriate training
        number_of_player_starts = len(scene.player_start_spots)
        player_start = self.rng.randint(0, max(0, number_of_player_starts - 1))
        # player_start = 140
        self.carla_client.start_episode(player_start)
        print(f'Starting new episode at {scene.map_name}, {player_start}...')

        # read and return status after reset
        measurements, image = self._read_data()

        return image

    def _read_data(self, ):
        """
        read data from carla
        :return: custom measurement data dict, camera image
        """
        measurements, sensor_data = self.carla_client.read_data()

        p_meas = measurements.player_measurements
        pos_x = p_meas.transform.location.x
        pos_y = p_meas.transform.location.y
        speed = p_meas.forward_speed * 3.6  # m/s -> km/h
        col_cars = p_meas.collision_vehicles
        col_ped = p_meas.collision_pedestrians
        col_other = p_meas.collision_other
        other_lane = 100 * p_meas.intersection_otherlane
        if other_lane:
            print('Intersection into other lane %.2f' % other_lane)
        offroad = 100 * p_meas.intersection_offroad
        if offroad:
            print('offroad %.2f' % offroad)
        agents_num = len(measurements.non_player_agents)
        distance = np.linalg.norm(np.array([pos_x, pos_y]) - self.target)
        meas = {
            'pos_x': pos_x,
            'pos_y': pos_y,
            'speed': speed,
            'col_damage': col_cars + col_ped + col_other,
            'other_lane': other_lane,
            'offroad': offroad,
            'agents_num': agents_num,
            'distance': distance,
            'autopilot_control': p_meas.autopilot_control
        }
        message = 'Vehicle at %.1f, %.1f, ' % (pos_x, pos_y)
        message += '%.0f km/h, ' % (speed,)
        message += 'Collision: vehicles=%.0f, pedestrians=%.0f, other=%.0f, ' % (col_cars, col_ped, col_other,)
        message += '%.0f%% other lane, %.0f%% off-road, ' % (other_lane, offroad,)
        message += '%d non-player agents in the scene.' % (agents_num,)
        # print(message)

        return meas, sensor_data['CameraRGB'].data.transpose([2, 0, 1])  # transpose into order (CHW)

    def _reward(self, pre_measurements, cur_measurements):
        """

        :param pre_measurements:
        :param cur_measurements:
        :return: reward
        """

        def delta(key):
            return cur_measurements[key] - pre_measurements[key]

        if pre_measurements is None:
            rwd = 0.0
        else:
            rwd = 0.05 * delta('speed') - 0.002 * delta('col_damage') \
                  - 2 * delta('offroad') - 2 * delta('other_lane')
            # print('delta speed %.3f' % delta('speed') )
            # print('[pre_offroad, current_offroad] =[%.2f, %.2f], reward: %.2f' %
            # (self.pre_measurements['offroad'], self.cur_measurements['offroad'], reward))

        # 1000 * delta('distance') +  ignore the distance for auto-pilot

        return rwd

    def _done(self, cur_measurements):
        """

        :param cur_measurements:
        :return:
        """
        # check distance to target
        d = cur_measurements['distance'] < 1  # final state arrived or not
        return d

    def close(self):
        """

        :return:
        """
        self.carla_client.disconnect()
        self.viewer.close()

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
