from pyaml import yaml
import argparse
import rl_learning as rl


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f)
    return config


def train_rl_agent(config, train=True):
    environment_params = config['environment_params']
    rl_params = config['rl_params']
    model_params = config['model_params']
    bandit_params = config['bandit_params']
    training_params = config['training_params']
    testing_params = config['testing_params']

    if train:
        rl.train_with_threads(environment_params, rl_params, bandit_params, model_params,
                              num_of_threads=training_params['number_of_threads'], epochs=training_params['epochs'], train=True,
                              display_state=False, use_processes=training_params['use_processes'])
    else:
        stat = rl.test_trained_model_with_random_play(environment_params, test_games=testing_params['test_games'], render=False)
        stat_for_display = rl.test_trained_model_with_random_play(environment_params, test_games=testing_params['display_games'],
                                                                  render=True)
        print stat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test the given environment')
    parser.add_argument('--config', dest='config_file', type=str, required=True,
                        help='full path to the yaml configuration file')
    parser.add_argument('--train', dest='train_and_test', action='store_true',
                        help='indicate whether to train and test the environment')
    parser.add_argument('--test', dest='only_test', action='store_true',
                        help='indicate whether simply test the environment with existing trained model')

    args = parser.parse_args()
    rl_config = load_config(args.config_file)
    if args.train_and_test:
        train_rl_agent(rl_config, train=args.train_and_test)

    if args.train_and_test or args.only_test:
        train_rl_agent(rl_config, train=False)
