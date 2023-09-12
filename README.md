# pyqt_gui
GUI for motility patterns

Certainly! Here's the README file content for you:


## Setting Up the Environment

To set up a new environment and install the required libraries, follow the steps below:

1. Create a new conda environment:
```bash
conda create -n your_env_name python=3.8
```
Replace `your_env_name` with a suitable name for your environment.

2. Activate the new environment:
```bash
conda activate your_env_name
```

3. Navigate to the directory containing the `requirements.txt` file:
```bash
cd /path/to/your/directory
```

4. Install the required libraries from `requirements.txt`:
```bash
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
```

Now, your environment is set up and ready to run the code.
