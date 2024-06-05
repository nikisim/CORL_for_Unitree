import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QHeaderView
from PyQt5.QtCore import QTimer
from PyQt5.Qt import QStandardItemModel, QStandardItem
import numpy as np
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QStyledItemDelegate

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())