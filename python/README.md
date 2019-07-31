### HornPy

Library for saving Keras models for future use in the Rust code.

##### How to use

Install the package:
```bash
pip install horn
```

Save Keras model using `save_model` function:
```python
from keras import Model
from horn import save_model

model = Model(name='mymodel')  # Keras model goes here
# Model is trained here

save_model(model, 'mymodel.horn')
```

The `*.horn` binary file is now the representation of your model.
 It includes the architecture and weights.
