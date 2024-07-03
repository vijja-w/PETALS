from evaluation.predictor import Predictor


config = Predictor.default_config
project_name = 'PETALS-Eval'

# specify the source expt_name and the method the source_expt used
config['method'] = 't'
config['source_expt_name'] = 't_1'

# eval expt_name
expt_name = f"eval_{config['source_expt_name']}"

# train t-prediction head
predictor = Predictor(config)
predictor.train(expt_name, project_name)

# get best model
predictor = Predictor.load(expt_name, project_name)
predictor.test(expt_name, project_name)
