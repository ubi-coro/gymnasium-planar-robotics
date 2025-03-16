import numpy as np

class ManualControl:
    ACCELERATION = 5.0

    def __init__(self) -> None:
        self.keys_pressed = set()
        self.reset_kinematics()

    def reset_kinematics(self):
        self.current_acc = np.array([0.0, 0.0], dtype=np.float64)

    # callback for key presses (add key to set)
    def on_key_press(self, key: str):
        self.keys_pressed.add(key)

    # callback for key releases (remove key from set)
    def on_key_release(self, key: str):
        self.keys_pressed.discard(key)

    def apply_key_kinematics(self):
        # apply dynamics based on currently pressed keys
        if 'up' in self.keys_pressed:
            self.current_acc[0] = -self.ACCELERATION
        elif 'down' in self.keys_pressed:
            self.current_acc[0] = self.ACCELERATION
        
        if 'left' in self.keys_pressed:
            self.current_acc[1] = -self.ACCELERATION
        elif 'right' in self.keys_pressed:
            self.current_acc[1] = self.ACCELERATION

    def get_action(self) -> np.ndarray:
        # apply dynamics based on currently pressed keys
        self.apply_key_kinematics()

        return self.current_acc.copy()
