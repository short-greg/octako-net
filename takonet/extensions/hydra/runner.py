import hydra
from . import base


class Runner(object):

    def __init__(self, cfg):

        self._cfg = cfg

    def run(self):

        experiment_cfg = list(self._cfg.blueprints.experiments.items())[0][1]

        override_base_params = hydra.utils.instantiate(experiment_cfg.override_base_params)
        override_trial_params = hydra.utils.instantiate(experiment_cfg.override_trial_params)

        study_builder: base.StudyBuilder = hydra.utils.instantiate(
            experiment_cfg.study_builder, override_base_params=override_base_params, 
            override_trial_params=override_trial_params
        )
        study: base.Study = hydra.utils.instantiate(experiment_cfg.study_type, study_builder=study_builder)

        study.run(experiment_cfg.name)

        # trial_param_dict = experiment_cfg.trial_params or {}
        # base_param_dict = experiment_cfg.base_params or {}

        # if experiment_cfg.experiment_type == "single":
        #     base_params = study.base_param_cls(**base_param_dict)
        #     study.run_single(experiment_cfg.name, experiment_cfg.base_params)
        # else:
        #     trial_params = study.trial_param_cls(**trial_param_dict)
        #     base_params = study.base_param_cls(**base_param_dict)
        #     study.run_study(
        #         experiment_cfg.name, experiment_cfg.n_studies, 
        #         base_params, trial_params
        #     )
