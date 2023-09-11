import sys
from PyQt5 import QtWidgets, QtCore
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tensorflow import keras
from IPython.display import display
from skimage.color import gray2rgb

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        #MARGIN = 40

        self.setWindowTitle("Text File Browser")
        self.setGeometry(100, 100, 1500, 1500)

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
        self.table_view.setGeometry(20, 60, 1500, 900)

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
        self.plot_clusters_button = QtWidgets.QPushButton("Cluster Plot", self)
        self.plot_clusters_button.clicked.connect(self.plot_clusters)
        self.plot_clusters_button.move(800, 20)
        
        self.run_model_button = QtWidgets.QPushButton("Run Model", self)
        self.run_model_button.clicked.connect(self.predict_classes)
        self.run_model_button.move(920, 20)  # Adjust the position as needed

        self.save_pickle_button = QtWidgets.QPushButton('Save df', self)
        self.save_pickle_button.clicked.connect(self.save_dataframe_as_pickle)
        #layout.addWidget(self.save_pickle_button)  # Assuming 'layout' is the layout variable you're using for your GUI
        self.save_pickle_button.move(1040,20)

        self.trajectory_plot_button = QtWidgets.QPushButton("Con. Plot", self)
        self.trajectory_plot_button.clicked.connect(self.plot_trajectory_colored)
        self.trajectory_plot_button.move(1160, 20)  # Adjust the position as needed

        # Add a button for plotting trajectories based on predicted classes
        self.class_plot_button = QtWidgets.QPushButton('Classes Plot', self)
        self.class_plot_button.clicked.connect(self.plot_predicted_classes)
        self.class_plot_button.move(1280, 20)

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
            #print("Motility DataFrame:")
            #print(self.motility_df)

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
            # size is not fixed
            plt.plot(row['centered_x'], row['centered_y'], '-', color='black')
            plt.axis('off')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            filename = os.path.join(output_dir, f"plot_{idx}.png")
            plt.savefig(filename, cmap='gray')
            plt.close()

            # Read the image as a grayscale numpy array
            image = plt.imread(filename)[:,:,0]  # Take one channel since it's grayscale
            # Convert the grayscale image to RGB
            rgb_image = gray2rgb(image)
            # Crop or resize if needed to make it 36x36 (since you mentioned the size is not fixed)
            if rgb_image.shape[0] != 36 or rgb_image.shape[1] != 36:
                rgb_image = rgb_image[:36, :36]
            image_arrays.append(rgb_image)

        # Add the list of numpy arrays as a new column to grouped_df
        self.grouped_df['image_arrays'] = image_arrays

        QtWidgets.QMessageBox.information(self, "Info", f"Plots saved in the '{output_dir}' directory and images added to the dataframe!")
        print(self.grouped_df['image_arrays'][0].shape)

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

    def predict_classes(self):
        # Ensure that the DataFrame exists and contains image arrays
        if self.grouped_df is None or 'image_arrays' not in self.grouped_df.columns:
            QtWidgets.QMessageBox.warning(self, "Warning", "Ensure both trajectory and motility files are loaded, merged, and images are generated!")
            return
    
        # Load the trained model
        model = keras.models.load_model("cnn_top50_classes6-0.h5", compile=False)
    
        # Prepare the images for the model (assuming grayscale images with 60x60 resolution)
        images = np.array(self.grouped_df['image_arrays'].tolist()).reshape(-1, 36, 36, 3)
    
        # Predict classes
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
    
        # Update DataFrame with predicted classes
        self.grouped_df['predicted_class'] = predicted_classes
    
        # Refresh the displayed dataframe to show the new 'predicted_class' column
        model_qt = pandasModel(self.grouped_df)
        self.table_view.setModel(model_qt)
    
    def save_dataframe_as_pickle(self):
        # Check if the dataframe exists
        if self.grouped_df is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "The dataframe is empty!")
            return
        
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Pickle File", "", "Pickle Files (*.pkl);;All Files (*)", options=options)
        
        if fileName:
            if not fileName.endswith('.pkl'):
                fileName += '.pkl'
            self.grouped_df.to_pickle(fileName)
            QtWidgets.QMessageBox.information(self, "Info", f"Dataframe saved as '{os.path.basename(fileName)}'!")

    def plot_trajectory_colored(self):
        # Check if grouped_df exists
        if self.grouped_df is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load and display the trajectory file first!")
            return

        # Extract unique concentrations and map them to colors
        unique_concentrations = self.grouped_df['concentration'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_concentrations)))  # Create a colormap that spans the unique concentrations
        concentration_to_color = dict(zip(unique_concentrations, colors))

        plt.figure(figsize=(10, 10))

        for concentration, color in concentration_to_color.items():
            subset = self.grouped_df[self.grouped_df['concentration'] == concentration]
            for idx, row in subset.iterrows():
                plt.plot(row['x'], row['y'], '-o', color=color, label=f"Concentration: {concentration}" if f"Concentration: {concentration}" not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title('Trajectories Colored by Concentration')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc="upper right")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_predicted_classes(self):
        # Check if grouped_df exists and contains cluster labels
        if self.grouped_df is None or 'predicted_class' not in self.grouped_df.columns:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load the trajectory file, generate the plots, add them to the dataframe, and run clustering first!")
            return

        # Generate a list of colors (extend this if more clusters are needed)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        plt.figure(figsize=(10, 10))
        
        # Create a set to keep track of classes that have been labeled already
        labeled_classes = set()

        for idx, row in self.grouped_df.iterrows():
            color = colors[row['predicted_class'] % len(colors)]  # Use modulo to avoid index out of range

            # Check if this class has been labeled already
            if row['predicted_class'] not in labeled_classes:
                plt.plot(row['x'], row['y'], '-o', color=color, label=row['predicted_class'])
                labeled_classes.add(row['predicted_class'])
            else:
                plt.plot(row['x'], row['y'], '-o', color=color)

        plt.title('Classified Trajectories')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
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
