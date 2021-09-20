from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import optuna
import typing
import math
from takonet.teaching import studies, dojos2 as dojos


PDELIM = "/"


class TrialSelector(ABC):

    def __init__(self, name, default):

        self.default = default
        self._name = name

    def select(self, path: str='', trial: optuna.Trial=None, best: dict=None):

        if best and self.cat_path(path) in best:
            result = self.update_best(best)
            return result
        elif trial is None:
            return self.default
        return self.suggest(path, trial)
    
    def cat_path(self, path: str=None, sub: str=None):
        full_path = self._name
        if path is not None:
            full_path = f'{path}/{full_path}'
        if sub is not None:
            full_path = f'{full_path}_{sub}' 
        return full_path

    @abstractmethod
    def suggest(self, path: str, trial: optuna.Trial):
        raise NotImplementedError
    
    @abstractmethod
    def _update_best_helper(self, best_val: dict, path: str):
        raise NotImplementedError

    def update_best(self, best_val: dict, path: str=None):
        if self.cat_path(path) not in best_val:
            return self.default
    
        return self._update_best_helper(best_val, path)

    @classmethod
    def from_dict(cls, params: dict):
        raise NotImplementedError
    

class Default(TrialSelector):

    def __init__(self, name: str, val):
        super().__init__(name, val)
        self._val = val
    
    def suggest(self, path: str, trial: optuna.Trial):
        return self._val
    
    def _update_best_helper(self, best_val):
        return self.default

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], params["val"])


class Int(TrialSelector):

    def __init__(self, name: str, low: int, high: int, default: int=0, base: int=None):
        super().__init__(name, default)

        self._low = low
        self._high = high
        self._base = base
    
    def suggest(self, path: str, trial: optuna.Trial):
        result = trial.suggest_int(
            self.cat_path(path) , self._low, self._high
        )
        if self._base is None:
            return result
        return self._base ** result

    def _update_best_helper(self, best_val: dict, path:str):
        val = best_val[self.cat_path(path)]
        if self._base is None:
            return val
        return self._base ** val

        # return int(math.log(val, self._base))

    @classmethod
    def from_dict(cls, params: dict):
        return cls(
            params['name'], 
            low=params['low'], high=params['high'], 
            base=params.get('base', None), default=params.get('default')
        )


class Bool(TrialSelector):

    def __init__(self, name: str, default: bool=True):
        super().__init__(name, default)

    def suggest(self, path: str, trial: optuna.Trial):

        return bool(trial.suggest_uniform(
            self.cat_path(path) , 0, 1
        ))

    def _update_best_helper(self, best_val: dict, path:str):
        val = best_val[self.cat_path(path)]
        return bool(val)

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], default=params.get('default'))


class Float(TrialSelector):

    def __init__(self, name: str, low: float=0., high: float=1., default: float=1.0):
        super().__init__(name, default)
        self._low = low
        self._high = high

    def suggest(self, path: str, trial: optuna.Trial):

        return trial.suggest_uniform(
            self.cat_path(path) , self._low, self._high
        )

    def _update_best_helper(self, best_val: dict, path:str):
        val = best_val[self.cat_path(path)]
        return val

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], default=params.get('default'))



class Categorical(TrialSelector):

    def __init__(self, name: str, categories: typing.List[str], default: str):
        super().__init__(name, default)
        self._categories = categories

    def suggest(self, path: str, trial: optuna.Trial):
        return trial.suggest_categorical(
            self.cat_path(path), self._categories
        )

    def _update_best_helper(self, best_val: dict, path:str):
        return best_val[self.cat_path(path)]

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], params["categories"], default=params.get('default'))


class ConditionalCategorical(TrialSelector):

    def __init__(self, name: str, categories: typing.Dict[str, str], default: str):
        super().__init__(name, default)
        self._categries = categories
    
    def _get_paths(self, path):
        base_path = self.cat_path(path)
        sub_path = self.cat_path(path, "sub")
        return base_path, sub_path
    
    def suggest(self, path: str, trial: optuna.Trial):
        base_path, sub_path = self._get_paths(path)
        base = trial.suggest_categorical(base_path, list(self._categries.keys()))
        sub_categories = self._categries[base]
        sub = trial.suggest_categorical(sub_path, sub_categories)
        return (base, sub)
    
    def _update_best_helper(self, best_val: dict, path:str):
        base_path, sub_path = self._get_paths(path)
        return best_val[base_path], best_val[sub_path] 

    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], params["categories"], default=params.get('default'))


class LogUniform(TrialSelector):

    def __init__(self, name: str, low: int, high: int, default: int):
        super().__init__(name, default)
        self._low = low
        self._high = high

    def suggest(self, path: str, trial: optuna.Trial):
        return trial.suggest_loguniform(
            self.cat_path(path), self._low, self._high
        )

    def _update_best_helper(self, best_val: dict, path:str):
        return best_val[self.cat_path(path)]
    
    @classmethod
    def from_dict(cls, params: dict):
        return cls(params['name'], params["low"], params["high"], default=params.get('default'))


class Non(object):

    @staticmethod
    def from_dict(params: dict):
        return params['value']


class Array(TrialSelector):

    def __init__(
        self, name: str, low: int, high: int, 
        params: typing.Dict[(str, TrialSelector)], 
        default=typing.List
    ):
        super().__init__(name, default)

        self._low = low
        self._high = high
        self._params = params

    def suggest(self, path: str, trial: optuna.Trial):
        params = []
        path = self.cat_path(path)
        
        size = trial.suggest_int(
            path + '/size',
            self._low,
            self._high
        )

        # TODO: HOW TO DEAL WITH THIS??
        for i in range(size):
            params.append({})
            for k, v in self._params.items():
                params[i][k] = v.suggest(
                    f'{path}/{i}', trial
                )
        return params

    def _get_param_by_name(self, name: str):

        for k, param in self._params.items():
            if param._name == name:
                return k, param

    def _update_best_helper(self, best_val: dict, path:str):
        best = best_val[self.cat_path(path)]
        
        size = best['size']
        result = []
        for i in range(size):
            i_str = str(i)
            result.append({})

            cur_params = best[i_str]
            for k in cur_params.keys():
                key, param = self._get_param_by_name(k)

                v = param.update_best(cur_params)
                result[i][key] = v
        
        return result
    
    @classmethod
    def from_dict(cls, params: dict):
        selectors: typing.Dict[str, TrialSelector] = {}
        
        for k, p in params:
            selectors[k] = ParamMap[p["type"]].from_dict(p)

        return cls(
            params['name'], low=params["low"], high=params["high"], params=selectors,
            default=params.get('default')
        )


ParamMap: typing.Dict[str, TrialSelector] = {
    "Array": Array,
    "Int": Int, 
    "LogUniform": LogUniform,
    "Categorical": Categorical,
    "ConditionalCategorical": ConditionalCategorical,
    "Bool": Bool,
    "Default": Default,
    "Non": Non
}

def convert_params(trial_params: dict):

    return {
        k: ParamMap[params['type']].from_dict(params)
        for k, params in trial_params.items()
    }

# TODO: Remove

class ParamConverter(object):

    def __init__(self, best_params: dict):

        self._best_params = best_params
    
    def to_dict(self):

        return self._nest_params(self._best_params)

    def _replace_with_lists(self):
        """[convert dictionaries in in the nested parameters 
        to lists where the param dictionary consists of the keys 'size' plus integers
        this is somewhat of a hack]

        Args:
            nested_params ([type]): [description]

        Returns:
            [type]: [description]
        """

        nested_params = self._best_params
        if 'size' in nested_params and len(nested_params) > 1:
            all_digits = True
            for k in nested_params:
                if k != 'size' and not k.isdigit():
                    all_digits = False
                    break
            result = []
            if all_digits is True:
                for k, v in nested_params.items():
                    if k == 'size': continue;
                    cur = int(k)
                    if len(result) >= cur:
                        result.extend([None] * (cur + 1 - len(result)))
                    result[cur] = self._replace_with_lists(v)
                return result

        for k, v in nested_params.items():
            if type(v) == dict:
                nested_params[k] = self._replace_with_lists(v)
            else:
                nested_params[k] = v

        return nested_params

    def _nest_params_helper(self, key_tuple, value, nested_params):
        
        cur = key_tuple[0]
        # if cur not in nested_params:
        #    nested_params[cur] = None

        if len(key_tuple) == 1:
            nested_params[cur] = value
        else:
            if cur not in nested_params:
                nested_params[cur] = {}
            self._nest_params_helper(
                key_tuple[1:], value, nested_params[cur]
            )

    def _nest_params(self, best_params):
        nested_params = {}
        for key, value in best_params.items():

            s = key.split(PDELIM)
            cur = s[0]
            # nested_params[cur] = 
            self._nest_params_helper(s, value, nested_params)

        return nested_params


@dataclass
class OptunaStudy(studies.Study):
    
    @abstractmethod
    def perform(self, trial=None, validation=False, best=None):
        pass


class StudyRunner(object):

    @staticmethod
    def get_direction(to_maximize):
        return optuna.study.StudyDirection.MAXIMIZE if to_maximize else optuna.study.StudyDirection.MINIMIZE

    def __init__(
        self, study: OptunaStudy, base_name: str, n_trials: int, to_maximize: bool
    ):
        self._study = study
        self._base_name = base_name
        self._n_trials = n_trials
        self._direction = self.get_direction(to_maximize)
    
    def get_objective(self, name: str, courses: typing.List[dojos.Course], parameters: typing.List) -> typing.Callable:
        cur: int = 0
        def objective(trial: optuna.Trial):
            nonlocal cur
            nonlocal courses
            nonlocal parameters
            course = self._study.perform(trial=trial, validation=True)
            parameters.append(trial.params)
            cur += 1
            courses.append(course)
            return course.evaluate()
        return objective

    def run(self, name) -> typing.List[dojos.Course]:

        classes: typing.List[dojos.Course] = []
        parameters = []
        study = optuna.create_study(direction=self._direction)
        objective = self.get_objective(name, classes, parameters)
        study.optimize(objective, self._n_trials)
        best_params = ParamConverter(study.best_params).to_dict()
        parameters.append(study.best_params)
        
        classes.append(self._study.perform(best=best_params))
        return classes, parameters
