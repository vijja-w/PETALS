from predictor import Predictor
from general.utils import clean_up

config = Predictor.default_config

project_name = 'PETALS'
seed_list = [1]
for seed in seed_list:
    config['seed'] = seed
    expt_name = f'vae_{seed}'
    # check_expt_name(expt_name, project_name)  # check if expt_name is taken
    print('')
    print('===========================================================================================================')
    print(f'Run expt_name: {expt_name}')
    print('===========================================================================================================')
    print('')

    # run
    predictor = Predictor(config)
    predictor.train(expt_name, project_name)

    # load best model
    predictor = Predictor.load(expt_name, project_name)
    predictor.test(expt_name, project_name)

    # terminate program
    clean_up()
