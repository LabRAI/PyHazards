Implementation
==============

PyHazard is built to be modular and extensible, allowing contributors to implement their own attack and defense strategies.
Below, we detail how to extend the framework by implementing custom attack and defense classes, with a focus on how to
leverage the provided dataset structure.

Dataset
-------

The ``Dataset`` class standardizes the data format across PyHazard. Here’s its structure:

.. code-block:: python

    class Dataset(object):
        def __init__(self, api_type='dgl', path='./data'):
            assert api_type in {'dgl', 'pyg'}, 'API type must be dgl or pyg'
            self.api_type = api_type
            self.path = path
            self.dataset_name = self.get_name()

            # DGLGraph or PyGData
            self.graph_dataset = None
            self.graph_data = None

            # meta data
            self.num_nodes = 0
            self.num_features = 0
            self.num_classes = 0

**Importance**: We are currently using the default ``api_type='pyg'`` to load the data. It is important to note that when
``api_type='pyg'``, ``self.graph_data`` should be an instance of ``torch_geometric.data.Data``. In your implementation, make
sure to use our defined Dataset class to build your code.

Device
------

To ensure consistency and simplicity when managing CUDA devices across attacks and defenses, we follow the convention
below:

- Both ``BaseAttack`` and ``BaseDefense`` define the device attribute ``self.device`` in their ``__init__()`` method.
- Subclasses should not manually redefine or modify the device logic.
- If you are implementing a custom attack or defense class, simply inherit from ``BaseAttack`` or ``BaseDefense``.
- You can directly access the device using: ``x = x.to(self.device)``

Implementing Attack
-------------------

To create a custom attack, you need to extend the abstract base class ``BaseAttack``. Here’s the structure
of ``BaseAttack``:

.. code-block:: python

    class BaseAttack(ABC):
        supported_api_types = set()
        supported_datasets = set()

        def __init__(self, dataset: Dataset, attack_node_fraction: float = None, model_path: str = None,
                     device: Optional[Union[str, torch.device]] = None):
            self.device = torch.device(device) if device else get_device()
            print(f"Using device: {self.device}")

            # graph data
            self.dataset = dataset
            self.graph_dataset = dataset.graph_dataset
            self.graph_data = dataset.graph_data

            # meta data
            self.num_nodes = dataset.num_nodes
            self.num_features = dataset.num_features
            self.num_classes = dataset.num_classes

            # params
            self.attack_node_fraction = attack_node_fraction
            self.model_path = model_path

            self._check_dataset_compatibility()

To implement your own attack:

1. **Inherit from ``BaseAttack``**:
   Create a new class that inherits from ``BaseAttack``. You’ll need to provide the following required parameters in the
   constructor:

   - ``dataset``: An instance of the ``Dataset`` class (see below for details).
   - ``attack_node_fraction``: A float between 0 and 1 representing the fraction of nodes to attack.
   - ``model_path`` (optional): A string specifying the path to a pre-trained model (defaults to ``None``).

   You need to implement the following methods:

   - ``attack()``: Add main attack logic here. If multiple attack types are supported, define the attack type as an optional
     argument to this function. For each specific attack type, implement a corresponding helper function such as
     ``_attack_type1()`` or ``_attack_type2()``, and call the appropriate helper inside ``attack()`` based on the given method name.
   - ``_load_model()``: Load victim model.
   - ``_train_target_model()``: Train victim model.
   - ``_train_attack_model()``: Train attack model.
   - ``_helper_func()`` (optional): Add your helper functions based on your needs, but keep the methods private.

2. **Implement the ``attack()`` Method**:
   Override the abstract ``attack()`` method with your attack logic, and return a dict of results. For example:

.. code-block:: python

    class MyCustomAttack(BaseAttack):
        supported_api_types = {"pyg"}  # "pyg" or "dgl"
        supported_datasets = {"Cora"}  # you can leave this blank if your method supports all datasets

        def __init__(self, dataset: Dataset, attack_node_fraction: float, model_path: str = None):
            super().__init__(dataset, attack_node_fraction, model_path)
            # Additional initialization if needed

        def attack(self):
            # Example: Access the graph and perform an attack
            print(f"Attacking {self.attack_node_fraction * 100}% of nodes")
            num_nodes = self.graph.num_nodes()
            print(f"Graph has {num_nodes} nodes")
            # Add your attack logic here
            return {
                'metric1': 'metric1 here',
                'metric2': 'metric2 here'
            }

        def _load_model(self):
            # add your logic here
            pass

        def _train_target_model(self):
            # add your logic here
            pass

        def _train_attack_model(self):
            # add your logic here
            pass

Implementing Defense
--------------------

To create a custom defense, you need to extend the abstract base class ``BaseDefense``. Here’s the structure
of ``BaseDefense``:

.. code-block:: python

    class BaseDefense(ABC):
        supported_api_types = set()
        supported_datasets = set()

        def __init__(self, dataset: Dataset, attack_node_fraction: float,
                     device: Optional[Union[str, torch.device]] = None):
            self.device = torch.device(device) if device else get_device()
            print(f"Using device: {self.device}")

            # graph data
            self.dataset = dataset
            self.graph_dataset = dataset.graph_dataset
            self.graph_data = dataset.graph_data

            # meta data
            self.num_nodes = dataset.num_nodes
            self.num_features = dataset.num_features
            self.num_classes = dataset.num_classes

            # params
            self.attack_node_fraction = attack_node_fraction

            self._check_dataset_compatibility()

To implement your own defense:

1. **Inherit from ``BaseDefense``**:
   Create a new class that inherits from ``BaseDefense``. You’ll need to provide the following required parameters in the
   constructor:

   - ``dataset``: An instance of the ``Dataset`` class (see below for details).
   - ``attack_node_fraction``: A float between 0 and 1 representing the fraction of nodes to attack.
   - ``model_path`` (optional): A string specifying the path to a pre-trained model (defaults to ``None``).

   You need to implement the following methods:

   - ``defense()``: Add main defense logic here. If multiple defense types are supported, define the defense type as an
     optional argument to this function. For each specific defense type, implement a corresponding helper function such as
     ``_defense_type1()`` or ``_defense_type2()``, and call the appropriate helper inside ``defense()``.
   - ``_load_model()``: Load victim model.
   - ``_train_target_model()``: Train victim model.
   - ``_train_defense_model()``: Train defense model.
   - ``_train_surrogate_model()``: Train attack model.
   - ``_helper_func()`` (optional): Add your helper functions based on your needs, but keep the methods private.

2. **Implement the ``defense()`` Method**:
   Override the abstract ``defense()`` method with your defense logic, and return a dict of results. For example:

.. code-block:: python

    class MyCustomDefense(BaseDefense):
        supported_api_types = {"pyg"}  # "pyg" or "dgl"
        supported_datasets = {"Cora"}  # you can leave this blank if your method supports all datasets

        def defend(self):
            # Step 1: Train target model
            target_model = self._train_target_model()
            # Step 2: Attack target model
            attack = MyCustomAttack(self.dataset, attack_node_fraction=0.3)
            attack.attack(target_model)
            # Step 3: Train defense model
            defense_model = self._train_defense_model()
            # Step 4: Test defense against attack
            attack = MyCustomAttack(self.dataset, attack_node_fraction=0.3)
            attack.attack(defense_model)
            # Print performance metrics

        def _load_model(self):
            # add your logic here
            pass

        def _train_target_model(self):
            # add your logic here
            pass

        def _train_defense_model(self):
            # add your logic here
            pass

        def _train_surrogate_model(self):
            # add your logic here
            pass

Miscellaneous Tips
------------------

- **Reference Implementation**: The ``ModelExtractionAttack0`` class is a fully implemented attack example. Study it for
  inspiration or as a template.
- **Flexibility**: Add as many helper functions as needed within your class to keep your code clean and modular.
- **Backbone Models**: We provide several basic backbone models like ``GCN, GraphSAGE``. You can use or add more
  at ``from models.nn import GraphSAGE``.
- **Example Scripts**: Please provide an example script in the ``examples/`` folder demonstrating how to run your code. This
  will significantly speed up our code review process.

By following these guidelines, you can seamlessly integrate your custom attack or defense strategies into PyHazard. Happy
coding!