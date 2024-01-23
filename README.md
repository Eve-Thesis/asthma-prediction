# Asthma prediction

Asthma prediction based on breathing sound using three pre-trained convolutional networks: VGGish, ResNet50, DenseNet121.

## Data Configuration

To set up the project data, you need to add a few files that are not included in the repository due to size or confidentiality reasons. Below is a list of these files and instructions on how to add them:

1. **data_params.py.temp** (in prediction/data/)
    - This file contains two parameters: `CSV_DATA_SAMPLES` and `DATA_SAMPLES`.
    - Modify the file name to `data_params.py`, removing `.temp` at the end.

2. **CSV_DATA_SAMPLES**
    - Add a CSV file containing data information of the samples for training the model.
    - Ensure that the CSV file has the following fields related to the data in the 'DATA_SAMPLES' folder:
        ```
        Uid;Folder Name;Breath filename;label;split
        ```
        - Set the path to this file in the `CSV_DATA_SAMPLES` variable in the `data_params.py` file.

3. **DATA_SAMPLES**
    - Add a file containing recordings of samples for training the model.
    - Set the path to this file in the `DATA_SAMPLES` variable in the `data_params.py` file.

4. **vggish_model.ckpt**
    - Add the file with weights `vggish_model.ckpt` for the [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model to the 'prediction/vggish' folder.

## Data Source

This project was developed using data from the University of Cambridge (project on COVID-19 diagnosis using cough, breath, and speech recordings). You can find the repository [here](https://github.com/cam-mobsys/covid19-sounds-neurips).
