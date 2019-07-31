import cv2
import numpy as np
import time
from threading import Thread

try:
    import pygame
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
    from pygame.locals import K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_0
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track

class HumanInterface():
    """
    Class to control a vehicle manually for debugging purposes
    """
    def __init__(self, parent):
        self.quit = False
        self._parent = parent
        self.WIDTH = 800
        self.HEIGHT = 600
        self.THROTTLE_DELTA = 0.05
        self.STEERING_DELTA = 0.01

        # pygame.init()
        # pygame.font.init()
        # self._clock = pygame.time.Clock()
        # self._display = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # pygame.display.set_caption("Human Agent")
        # self.font = font = pygame.font.Font(pygame.font.get_default_font(), 15)

    def run(self):
        while not self._parent.agent_engaged and not self.quit:
            time.sleep(0.5)

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")
        self.font = pygame.font.Font(pygame.font.get_default_font(), 15)
        controller = KeyboardControl()
        while not self.quit:
            print('running')
            self._clock.tick_busy_loop(20)
            controller.parse_events(self._parent.current_control, self._clock)
            # Process events
            pygame.event.pump()

            # process sensor data
            mode = controller.displaymode
            input_data = self._parent.sensor_interface.get_data()
            
            if mode == 1:
                image = input_data['Center'][1][:,:,-2::-1]
            elif mode == 2:
                image = input_data['Left'][1][:,:,-2::-1]
            elif mode == 3:
                image = input_data['Right'][1][:,:,-2::-1]
            elif mode == 4:
                image = input_data['Rear'][1][:,:,-2::-1]
            elif 5 <= mode <= 7:
                image = input_data['Depth'][1][:,:,-2::-1]
                if mode != 5:
                    R, G, B = cv2.split(image.astype('float32'))
                    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
                    if mode == 6:
                        gray = (normalized * 255).astype('uint8')
                    else:
                        in_meter = normalized * 1000
                        logscale = np.log10(in_meter + 1 - np.min(in_meter))
                        divisor = np.max(logscale) / 255.0
                        gray = (logscale / divisor).astype('uint8')
                    image = cv2.merge((gray, gray, gray))

            elif mode == 8:
                points = input_data['LIDAR'][1]
                lidar_data = np.array(points[:, :2])
                lidar_data *= min(self.HEIGHT, self.WIDTH) / 100.0
                lidar_data += (0.5 * self.HEIGHT, 0.5 * self.WIDTH)
                lidar_data = np.fabs(lidar_data)
                lidar_data = lidar_data.astype(np.int32)
                lidar_data = np.reshape(lidar_data, (-1, 2))
                lidar_img_size = (self.HEIGHT, self.WIDTH, 3)
                image = np.zeros(lidar_img_size)    
                image[tuple(lidar_data.T)] = (255, 255, 255)
                image = image[:,::-1]

            # display image
            self._surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            if self._surface is not None:
                self._display.blit(self._surface, (0, 0))

            text = ('Center', 'Left', 'Right', 'Rear', 'Depth (RAW)', 'Depth (Gray Scale)', 'Depth (Logarithmic Gray Scale)', 'LIDAR', '')[mode-1]
            textsurface = self.font.render(text, False, (255, 255, 255))
            self._display.blit(textsurface, (400, 550))
            text = 'GPS : {}'.format(input_data['GPS'][1])
            textsurface = self.font.render(text, False, (255, 255, 255))
            self._display.blit(textsurface, (0, 0))
            text = 'Score : {}'.format(input_data['Score'][1])
            textsurface = self.font.render(text, False, (255, 255, 255))
            self._display.blit(textsurface, (0, 20))
            
            pygame.display.flip()

        pygame.quit()


class HumanAgent2(AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = Track.ALL_SENSORS_HDMAP_WAYPOINTS

        self.agent_engaged = False
        self.current_control = carla.VehicleControl()
        self.current_control.steer = 0.0
        self.current_control.throttle = 1.0
        self.current_control.brake = 0.0
        self.current_control.hand_brake = False
        self._hic = HumanInterface(self)
        self._thread = Thread(target=self._hic.run)
        self._thread.start()


    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            ['sensor.camera.rgb', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                   'yaw': yaw, 'pitch': pitch, 'roll': roll,
                                   'width': width, 'height': height, 'fov': fov}, 'Sensor01'],
            ['sensor.camera.rgb', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                   'yaw': yaw, 'pitch': pitch, 'roll': roll,
                                   'width': width, 'height': height, 'fov': fov}, 'Sensor02'],

            ['sensor.lidar.ray_cast', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                       'yaw': yaw, 'pitch': pitch, 'roll': roll}, 'Sensor03']
        ]

        """
        sensors = [{'type': 'sensor.camera.rgb', 'x':0.7, 'y':0.0, 'z':1.60, 'roll':0.0, 'pitch':0.0, 'yaw':0.0,
                    'width':800, 'height':600, 'fov':100, 'id': 'Center'},

                   {'type': 'sensor.camera.rgb', 'x':0.7, 'y':-0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
                    'yaw': -45.0, 'width': 800, 'height': 600, 'fov': 100, 'id': 'Left'},

                   {'type': 'sensor.camera.rgb', 'x': 0.7, 'y':0.4, 'z':1.60, 'roll':0.0, 'pitch':0.0, 'yaw':45.0,
                    'width':800, 'height':600, 'fov': 100, 'id': 'Right'},

                   {'type': 'sensor.camera.rgb', 'x': -1.8, 'y': 0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
                    'yaw': 180.0, 'width': 800, 'height': 600, 'fov': 130, 'id': 'Rear'},

                   {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},

                   {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
                    'yaw': -45.0, 'id': 'LIDAR'},

                   {'type': 'sensor.score', 'reading_frequency': 10, 'id': 'Score'},

                   {'type': 'sensor.camera.depth', 'x':0.7, 'y':0.0, 'z':1.60, 'roll':0.0, 'pitch':0.0, 'yaw':0.0,
                    'width':800, 'height':600, 'fov':100, 'id': 'Depth'},
                  ]

        return sensors

    def run_step(self, input_data, timestamp):
        self.agent_engaged = True
        time.sleep(0.1)
        return self.current_control

    def destroy(self):
        self._hic.quit = True
        self._thread.join()


class KeyboardControl(object):
    def __init__(self):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self.displaymode = 1

    def parse_events(self, control, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            control.steer = self._control.steer
            control.throttle = self._control.throttle
            control.brake = self._control.brake
            control.hand_brake = self._control.hand_brake

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_1]:
            self.displaymode = 1
        elif keys[K_2]:
            self.displaymode = 2
        elif keys[K_3]:
            self.displaymode = 3
        elif keys[K_4]:
            self.displaymode = 4
        elif keys[K_5]:
            self.displaymode = 5
        elif keys[K_6]:
            self.displaymode = 6
        elif keys[K_7]:
            self.displaymode = 7
        elif keys[K_8]:
            self.displaymode = 8

        self._control.throttle = 0.6 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 15.0 * 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.95, max(-0.95, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]