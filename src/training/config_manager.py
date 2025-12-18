# a class that loads the configurations for training a dl model
import yaml
from typing import Any, Dict, List, Union, Optional

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[dict] = None):
        assert config_path is not None or config_dict is not None, 'at least one of config_path and config_dict should be given'

        if config_path is not None:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        elif config_dict is not None:
            self.config = config_dict
        
        # Store hyperparameter configurations separately. this should be something like this in the yaml file:
        '''
        hyperparameters:
            learning_rate:
                min: 0.000001
                max: 0.1
            batch_size:
                values: [64, 128, 256]
        '''
        self.hyperparameters = {}
        if 'hyperparameters' in self.config:
            self.hyperparameters = self.config['hyperparameters']
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a regular configuration value"""
        if '.' in key_path:
            keys = key_path.split('.') # for instnace if the key is train.epoch, it will be [train, epoch]
        else:
            keys = [key_path]

        value = self.config
        try:
            for key in keys: # will return config[train][epoch]
                value = value[key]
            return value
        except KeyError:
            print(f'{key} was not found in the hyperparam manager')
            return default
    
    def get_hyper(self, param_name: str) -> Dict:
        """Get hyperparameter configuration including its type and range/values"""
        if param_name not in self.hyperparameters:
            raise KeyError(f"Hyperparameter '{param_name}' not found in config")
        
        param_config = self.hyperparameters[param_name]
        
        # Determine parameter type and return appropriate configuration
        if 'values' in param_config:
            return {
                'type': 'categorical',
                'values': param_config['values']
            }
        elif 'min' in param_config and 'max' in param_config:
            return {
                'type': 'continuous',
                'min': param_config['min'],
                'max': param_config['max']
            }
        else:
            raise ValueError(f"Invalid hyperparameter configuration for '{param_name}'")
    
    def get_all_hyperparameters(self) -> Dict[str, Dict]:
        """Get all hyperparameter configurations"""
        return {name: self.get_hyper(name) for name in self.hyperparameters}
    
    def add_hyperparam(self, key, value):
        self.config[key] = value
    
    def create_bayes_opt_config(self):
        """Create configuration for Bayesian Optimization"""
        pbounds = {}
        param_types = {}
        
        for param_name, param_config in self.hyperparameters.items():
            if 'values' in param_config:
                # Categorical parameter: use index between 0 and 0.999
                pbounds[f"{param_name}_idx"] = (0, 0.999)
                param_types[param_name] = {
                    'type': 'categorical',
                    'values': param_config['values']
                }
            elif 'min' in param_config and 'max' in param_config:
                # Continuous parameter: use actual range
                pbounds[param_name] = (param_config['min'], param_config['max'])
                param_types[param_name] = {
                    'type': 'continuous'
                }
        
        return pbounds, param_types
    
    def create_trial_config(self, trial_params: Dict[str, float], use_train = True) -> Dict[str, Any]:
        """
        Convert Bayesian Optimization parameters to actual configuration values
        
        Args:
            trial_params: Dictionary of parameters from Bayesian Optimization
            use_train: Add a train. before each hyperparam (for compatibillity reasons in the code)
        
        Returns:
            Dictionary with actual parameter values to use in training
        """
        actual_params = {}
        
        for param_name, param_config in self.hyperparameters.items():
            if 'values' in param_config:
                # Convert index back to categorical value
                idx_param = f"{param_name}_idx"
                if idx_param in trial_params:
                    values = param_config['values']
                    idx = int(trial_params[idx_param] * len(values))
                    if use_train:
                        actual_params['train.' + param_name] = values[idx]
                    else:
                        actual_params[param_name] = values[idx]
            else:
                # Use continuous parameter directly
                if param_name in trial_params:
                    if use_train:
                        actual_params['train.' + param_name] = trial_params[param_name]
                    else:
                        actual_params[param_name] = trial_params[param_name]
        
        return actual_params

# Example usage:
if __name__ == "__main__":
    # Example configuration
    example_config = {
        'model': {
            'name': 'RF'
        },
        'train': {
            'num_epochs': 150
        },
        'hyperparameters': {
            'learning_rate': {
                'min': 0.000001,
                'max': 0.1
            },
            'batch_size': {
                'values': [64, 128, 256]
            },
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            'lr_scheduler': {
                'values': ['none', 'default', 'steplr', 'reducelronplateau']
            }
        }
    }
    
    # Create config manager
    cm = ConfigManager(config_dict=example_config)
    
    # Get regular config value
    print("Model name:", cm.get('model.name'))
    
    # Get hyperparameter config
    print("\nLearning rate config:", cm.get_hyper('learning_rate'))
    print("Batch size options:", cm.get_hyper('batch_size'))
    
    # Create Bayesian Optimization config
    pbounds, param_types = cm.create_bayes_opt_config()
    print("\nBayesian Optimization bounds:", pbounds)
    
    # Example of converting trial parameters to actual values
    trial_params = {
        'learning_rate': 0.001,
        'batch_size_idx': 0.4,
        'optimizer_idx': 0.7,
        'lr_scheduler_idx': 0.2
    }
    actual_params = cm.create_trial_config(trial_params)
    print("\nConverted parameters:", actual_params)