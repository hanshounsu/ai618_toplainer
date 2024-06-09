import yaml
import argparse
from termcolor import colored


class ConfigXT(object):
    def __init__(self, config_files=None):
        parser = None
        if config_files is None:
            parser = argparse.ArgumentParser() # 터미널에 입력해준 arg 인식
            parser.add_argument('-c', '--config', nargs="*", type=str,
                                default=['./configs/diffusion_unet.yml'])
            # namespace : Bring known args (in tuple),  _ : bring unknown args (in list)
            namespace, _ = parser.parse_known_args()
            config_files = namespace.config

        for c in config_files:
            self.load(c)
        self.parse(parser)

    def load(self, filename):
        yaml_file = open(filename, 'r')
        yaml_dict = yaml.safe_load(yaml_file)

        vars(self).update(yaml_dict)

    def save(self, filename=None, verbose=False):
        yaml_dict = dict()
        for var in vars(self):
            if var not in ['config', 'verbose']:
                value = getattr(self, var)
                yaml_dict[var] = value

        with open(filename, 'w') as yaml_file:
            yaml.dump(yaml_dict, yaml_file, sort_keys=False)
            if verbose:
                print('Configuration file is saved to \'%s\'' % (filename))

    def parse(self, parser):
        if parser is not None:
            for var in vars(self):  # retrieve keys of dict
                value = getattr(self, var)  # retrive values of dict
                argument = '--' + var
                if type(value) is list:
                    parser.add_argument(argument, nargs="*",
                                        type=type(value[0]), default=value)
                else:
                    parser.add_argument(
                        argument, type=type(value), default=value)

            parser.add_argument('-v', '--verbose', action='store_true')
            args = parser.parse_args()
            self.verbose(args.verbose)

            for var in vars(args):
                if var not in ['c', 'v']:  # c : config, v : verbose
                    # add dict value of key=var, value=getattr(args, var)
                    setattr(self, var, getattr(args, var))
    
    def add(self, var, value):
        setattr(self, var, value)
        

    def verbose(self, v):
        if v:
            print(colored('[ Configurations ]', 'green'))
            for var in vars(self):
                value = getattr(self, var)
                print(colored('| ', 'green') +
                      colored(var, 'blue') + ': ' + str(value))

            print('\n')


def main():
    config = ConfigXT()


if __name__ == "__main__":
    main()
