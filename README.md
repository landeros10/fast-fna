# FAST-FNA

A deep learning project for training and evaluating neural networks with custom data structures and 
layers.

## Project Structure

```
├── align
│   ├── data_structures.py  # Defines custom data structures used in the project
│
├── ml
│   ├── layers.py           # Implements custom neural network layers
│   ├── models.py           # Defines model architectures using custom layers
│   ├── train.py           # Handles model training and evaluation
│   ├── util.py            # Utility functions for data processing and training
```

## Installation

Ensure you have Python installed along with the required dependencies. You can install them using:

```sh
pip install -r requirements.txt
```

## Usage

1. **Define your dataset and data structures** in `align/data_structures.py`.
2. **Customize or create neural network layers** in `ml/layers.py`.
3. **Build and modify your model architecture** in `ml/models.py`.
4. **Train and evaluate models** using `ml/train.py`.
5. **Use utility functions** in `ml/util.py` for preprocessing, logging, and evaluation.

### Training a Model

Run the following command to start training:

```sh
python ml/train.py --config config.yaml
```

Modify `config.yaml` to change hyperparameters, dataset paths, or model settings.

## Contributing

Feel free to fork the repository and submit pull requests with improvements or fixes.

## License

[MIT License](LICENSE)


