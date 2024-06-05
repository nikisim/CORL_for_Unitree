import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QHeaderView
from PyQt5.QtCore import QTimer
from PyQt5.Qt import QStandardItemModel, QStandardItem
import numpy as np
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QStyledItemDelegate
from PyQt5.QtCore import pyqtSignal
import threading
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import QThread, QObject, pyqtSignal

import jax
import optax
import flax
from algorithms.offline.rebrac_Fetch_UR5 import DetActor, ActorTrainState, ReplayBuffer, Config
import gym
import gym_UR5_FetchReach
from algorithms.offline.rebrac_Fetch_UR5 import wrap_env, evaluate, compute_mean_std
import json
import argparse
import numpy as np
import tkinter as tk
from threading import Thread

class Worker(QObject):
    update_signal = pyqtSignal(dict)

    # Load the TrainState from a file
    def load_train_state(self,save_path, state_structure):
        with open(save_path, 'rb') as f:
            state_dict = flax.serialization.from_bytes(state_structure, f.read())
        return state_dict

    def create_train_state(self,actor_module, actor_key, init_state, actor_learning_rate):
        return ActorTrainState.create(
            apply_fn=actor_module.apply,
            params=actor_module.init(actor_key, init_state),
            target_params=actor_module.init(actor_key, init_state),
            tx=optax.adam(learning_rate=actor_learning_rate),
        )

    def get_action(self, obs):

        config_path = '/home/nikisim/Mag_diplom/CORL/data/saved_models/sim_as_real/FetchReach_UR5_ReBRAC_ac1300_bc_1100_ReBRAC_1300.0_1100.0_Reach--d8e5f605/config.json'
        model_path = '/home/nikisim/Mag_diplom/CORL/data/saved_models/sim_as_real/FetchReach_UR5_ReBRAC_ac1300_bc_1100_ReBRAC_1300.0_1100.0_Reach--d8e5f605/actor_state4200.pkl'

        with open(config_path) as json_file:
            config_dict = json.load(json_file)

        config = Config.from_dict(Config, config_dict)

        dataset_name = '/home/nikisim/Mag_diplom/UR5_FetchReach_real/datasets/UR5_FetchReach_real_small.npy'

        buffer = ReplayBuffer()
        buffer.create_from_d4rl(
            dataset_name, False, False
        )
        @jax.jit
        def actor_action_fn(params: jax.Array, obs: jax.Array):
            return actor.apply_fn(params, obs)

        init_state = buffer.data["states"][0][None, ...]
        init_action = buffer.data["actions"][0][None, ...]

        key = jax.random.PRNGKey(seed=51)
        key, actor_key, _ = jax.random.split(key, 3)

        actor_module = DetActor(
                action_dim=init_action.shape[-1],
                hidden_dim=config.hidden_dim,
                layernorm=config.actor_ln,
                n_hiddens=config.actor_n_hiddens,
            )

        train_state_struc = self.create_train_state(actor_module, actor_key, init_state, config.actor_learning_rate)

        actor = self.load_train_state(model_path, train_state_struc)

        action = np.asarray(jax.device_get(actor_action_fn(actor.params, obs)))

        return action

    def run_simulation(self):
        # Your robot simulation code here...
        # Instead of directly updating the GUI, emit a signal:

        env = gym.make('gym_UR5_FetchReach/UR5_FetchReachEnv-v0', render=True)
        # env = gym.make('FetchPickAndPlaceDense-v2', render_mode='human')

        dataset_name = '/home/nikisim/Mag_diplom/UR5_FetchReach_real/datasets/UR5_FetchReach_real_small.npy'

        buffer = ReplayBuffer()
        buffer.create_from_d4rl(
            dataset_name, False, False
        )

        env.action_space.seed(42)
        env.observation_space.seed(42)
        env = wrap_env(env, buffer.mean, buffer.std)

        # for _ in range(5):
        obs, _ = env.reset()
        done = False
        # print("Observation:", obs)
        # print("Observation_type:", type(obs))
        total_reward = 0.0
        while not done:
            action = self.get_action(obs)
            obs, reward, termination, truncation, info = env.step(action)
            print("EE_POS", obs[:3])
            print("TARGET_POS", obs[7:10])
            error = -reward
            print("Error", -reward)

            new_row_data = {
                    'ee_pos': obs[:3],
                    'target_pos': obs[7:10],
                    'error': error,
                    'action': action
                }
            
            self.update_signal.emit(new_row_data)

            done = termination or truncation

            total_reward += reward

class PandasTableModel(QStandardItemModel):
    def __init__(self, data):
        QStandardItemModel.__init__(self)
        self._data = data
        for row in data.iterrows():
            data_row = []
            for item in row[1]:
                cell = QStandardItem(str(item))
                data_row.append(cell)
            self.appendRow(data_row)
            
    def update_data(self, data):
        self.clear()
        self._data = data
        for row in data.iterrows():
            data_row = []
            for item in row[1]:
                cell = QStandardItem(str(item))
                data_row.append(cell)
            self.appendRow(data_row)
        
class CustomDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.column() == 2:  # Assuming 'error' column is at index 2
            error_value = float(index.data())
            if error_value < 0.05:
                option.backgroundBrush = QColor('green')


class MainWindow(QMainWindow):
    new_data_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.table = QTableView(self)
        self.model = PandasTableModel(pd.DataFrame(columns=['ee_pos', 'target_pos', 'error', 'action']))
        self.table.setModel(self.model)
        
        # Set column names
        # Set column names for the table view
        self.model.setHorizontalHeaderLabels(['ee_pos', 'target_pos', 'error', 'action'])
        # Set up a timer to refresh the DataFrame content periodically
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # Update interval in milliseconds
        self.timer.timeout.connect(self.update_data)
        self.timer.start()

        # Set the custom delegate for cell coloring
        delegate = CustomDelegate()
        self.table.setItemDelegate(delegate)
        
        self.setCentralWidget(self.table)
        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle('Robot Data')

        # self.new_data_signal.connect(self.update_data_from_external)
        self.robot_simulation_thread = RobotSimulationThread()
        self.robot_simulation_thread.worker.update_signal.connect(self.new_data_signal.emit)
        self.robot_simulation_thread.start()

    # def update_data(self):
    #     # Here, you would fetch the latest data from your robot and update the DataFrame
    #     # For demonstration, let's just add a new row with random data
    #     new_row = {'ee_pos': [0.1, 0.2, 0.3], 'target_pos': [0.4, 0.5, 0.6], 'error': 0.05, 'action': 'Move Up'}
    #     df = pd.concat([self.model._data, pd.DataFrame([new_row])], ignore_index=True)
    #     self.model.update_data(df)
    
    def update_data(self):
        # Fetch the latest data from your robot
        new_row = {
            'ee_pos': [0.1, 0.2, 0.3],
            'target_pos': [0.4, 0.5, 0.6],
            'error': np.round(np.random.uniform(0.0, 0.3),3),  # Initial error value
            'action': 'Move Up'
        }

        df = self.model._data
        if len(df) >= 5:
            return
        if not df.empty and df['error'].iloc[-1] > 0.05:
            # Update the error value of the last row until it is less than 0.05
            df.loc[df.index[-1], :] = new_row
            self.model.update_data(df)
        else:
            # Add a new row with the updated values
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            self.model.update_data(df)

    def update_data_from_external(self, new_row_data):
        # Update the DataFrame with the new row data from external sources
        df = self.model._data
        if len(df) >= 5:
            return
        if not df.empty and df['error'].iloc[-1] > 0.05:
            # Update the error value of the last row until it is less than 0.05
            df.loc[df.index[-1], :] = new_row_data
            self.model.update_data(df)
        else:
            # Add a new row with the updated values
            df = pd.concat([df, pd.DataFrame([new_row_data])], ignore_index=True)
            self.model.update_data(df)

class RobotSimulationThread(QObject):
    update_signal = pyqtSignal(dict)  # Signal to send updates

    def __init__(self):
        super().__init__()
        self.worker = Worker()
        self.worker.moveToThread(self)

    def run(self):
        self.worker.run_simulation()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())