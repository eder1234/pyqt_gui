import sys
from PyQt5 import QtWidgets, QtCore
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Text File Browser")
        self.setGeometry(100, 100, 1000, 1000)

        # Create a button to browse trajectory file
        self.trajectory_button = QtWidgets.QPushButton("Trajectory file", self)
        self.trajectory_button.clicked.connect(self.load_trajectory_file)
        self.trajectory_button.move(20, 20)

        # Create a button to browse motility file
        self.motility_button = QtWidgets.QPushButton("Motility file", self)
        self.motility_button.clicked.connect(self.load_motility_file)
        self.motility_button.move(150, 20)

        # Create a button to merge files
        self.merge_button = QtWidgets.QPushButton("Merge files", self)
        self.merge_button.clicked.connect(self.merge_files)
        self.merge_button.move(280, 20)

        # Create table view to display DataFrame
        self.table_view = QtWidgets.QTableView(self)
        self.table_view.setGeometry(20, 60, 900, 900)

        self.trajectory_df = None
        self.motility_df = None
        self.grouped_df = None

        # Create a button to generate plots
        self.plot_button = QtWidgets.QPushButton("Generate imgs", self)
        self.plot_button.clicked.connect(self.generate_plots)
        self.plot_button.move(410, 20)

        # Create a spin box for selecting the number of clusters
        self.cluster_spinbox = QtWidgets.QSpinBox(self)
        self.cluster_spinbox.setMinimum(2)  # Minimum 2 clusters
        self.cluster_spinbox.setValue(5)    # Default value
        self.cluster_spinbox.move(550, 20)

        # Create a button to run the clustering
        self.cluster_button = QtWidgets.QPushButton("Run AH", self)
        self.cluster_button.clicked.connect(self.run_clustering)
        self.cluster_button.move(680, 20)

        # Create a button to display the cluster plot
        self.plot_clusters_button = QtWidgets.QPushButton("Plot Clusters", self)
        self.plot_clusters_button.clicked.connect(self.plot_clusters)
        self.plot_clusters_button.move(800, 20)

        self.show()

    def load_trajectory_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open trajectory file", "", "Text Files (*.txt);;CSV Files (*.csv)")
        if file_path:
            # Read CSV file into DataFrame
            columns = ["name", "date", "concentration", "exposure", "id", "x", "y"]
            self.trajectory_df = pd.read_csv(file_path, header=None, names=columns)
            # Group and display
            self.display_trajectory()

    def display_trajectory(self):
        self.grouped_df = self.trajectory_df.groupby(["name", "date", "concentration", "exposure", "id"]).agg(
            x=pd.NamedAgg(column='x', aggfunc=list),
            y=pd.NamedAgg(column='y', aggfunc=list)
        ).reset_index()

        # Center the x and y coordinates
        self.grouped_df['centered_x'] = self.grouped_df.apply(lambda row: [x - sum(row['x'])/len(row['x']) for x in row['x']], axis=1)
        self.grouped_df['centered_y'] = self.grouped_df.apply(lambda row: [y - sum(row['y'])/len(row['y']) for y in row['y']], axis=1)

        # Convert DataFrame to Qt Table Model and display
        model = pandasModel(self.grouped_df)
        self.table_view.setModel(model)

    def load_motility_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open motility file", "", "Text Files (*.txt);;CSV Files (*.csv)")
        if file_path:
            # Read the motility file with custom column names
            columns = ["name", "date", "exposure", "concentration", "VCL", 'VAP', 'VSL', 'LIN', 'STR', 'WOB', 'BeatCross', 'ALH']
            self.motility_df = pd.read_csv(file_path, header=None, names=columns, skiprows=1)

            # Print the motility DataFrame to the terminal
            print("Motility DataFrame:")
            print(self.motility_df)

    def merge_files(self):
        if self.grouped_df is not None and self.motility_df is not None:
            # Extract the specified columns from motility_df
            motility_columns = ["VCL", "VAP", "VSL", "LIN", "STR", "WOB", "BeatCross", "ALH"]
            extracted_motility = self.motility_df[motility_columns]

            # Merge grouped_df with the extracted columns by index
            merged_df = pd.concat([self.grouped_df, extracted_motility], axis=1)

            # Display the merged DataFrame
            model = pandasModel(merged_df)
            self.table_view.setModel(model)

    def generate_plots(self):
        # Check if grouped_df exists
        if self.grouped_df is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load and display the trajectory file first!")
            return

        # Create a directory to store the plots
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_arrays = []

        # Generate plots for each element in grouped_df
        for idx, row in self.grouped_df.iterrows():
            plt.figure(figsize=(1, 1), dpi=60)  # 1x1 inch figure at 60 dpi results in 60x60 pixel image
            plt.plot(row['centered_x'], row['centered_y'], '-o', color='black')
            plt.axis('off')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            filename = os.path.join(output_dir, f"plot_{idx}.png")
            plt.savefig(filename, cmap='gray')
            plt.close()

            # Read the image as a numpy array and append to the list
            image = plt.imread(filename)
            image_arrays.append(image)

        # Add the list of numpy arrays as a new column to grouped_df
        self.grouped_df['image_arrays'] = image_arrays

        QtWidgets.QMessageBox.information(self, "Info", f"Plots saved in the '{output_dir}' directory and images added to the dataframe!")

    def run_clustering(self):
        # Check if grouped_df exists and contains image_arrays
        if self.grouped_df is None or 'image_arrays' not in self.grouped_df.columns:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load the trajectory file, generate the plots, and add them to the dataframe first!")
            return

        # Flatten the image arrays for clustering
        flattened_images = [img.flatten() for img in self.grouped_df['image_arrays']]

        # Number of clusters from the spinbox
        n_clusters = self.cluster_spinbox.value()

        # Run agglomerative clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(flattened_images)

        # Add cluster labels to the dataframe
        self.grouped_df['cluster'] = cluster_labels

        # Refresh the displayed dataframe to show the new cluster column
        model = pandasModel(self.grouped_df)
        self.table_view.setModel(model)

    def plot_clusters(self):
        # Check if grouped_df exists and contains cluster labels
        if self.grouped_df is None or 'cluster' not in self.grouped_df.columns:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load the trajectory file, generate the plots, add them to the dataframe, and run clustering first!")
            return

        # Generate a list of colors (extend this if more clusters are needed)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        plt.figure(figsize=(10, 10))

        for idx, row in self.grouped_df.iterrows():
            color = colors[row['cluster'] % len(colors)]  # Use modulo to avoid index out of range
            plt.plot(row['x'], row['y'], '-o', color=color)

        list_cluster = self.grouped_df['cluster'].unique()
        plt.title('Clustered Trajectories')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(list_cluster)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


class pandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._df.columns[col]
        return None
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
