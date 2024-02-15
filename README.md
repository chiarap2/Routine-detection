# (Non-) Routine Behaviour Detection

This repository contains the code for identifying routine and non-routine behavior from a dataset of trajectories. The trajectories are first summarized using MAT-Sum, a method for summarizing trajectories while preserving their semantic quality.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running the main script, you need to install the required packages listed in the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## Configuration

The `main.py` script takes as input a `config.json` file where you need to specify the paths of the files. An example of a `config.json` file is:

```bash
{
    "data": {
        "trajectories": <path_to_trajectory_data>,
        "tiles": <path_to_bbox_data>,
        "poi": <path_to_POI_data>,
        "landuse": <path_to_landuse_data>,
        "pt": <path_to_public_transportation_data>
    }
}
```
Replace `<path_to_trajectory_data>` and similar with the paths to your trajectory data file and the desired output location, respectively.

## Running the Script

After setting up the prerequisites and configuration, you can run the `main.py` script as follows:
```
python main.py config.json
```

