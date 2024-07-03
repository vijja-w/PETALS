from predictor import Predictor

config = Predictor.default_config

project_name = 'PETALS'
seed_list = [1]
for seed in seed_list:
    config['seed'] = seed
    expt_name = f'dr_{seed}'
    print('')
    print('===========================================================================================================')
    print(f'Run expt_name: {expt_name}')
    print('===========================================================================================================')
    print('')

    # run
    predictor = Predictor(config)
    predictor.train(expt_name)

