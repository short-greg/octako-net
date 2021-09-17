# import torch
# import typing


# class StudyItem(object):

#     def __init__(self, item_dict: typing.Dict[str, torch.Tensor]):
#         """[summary]

#         Args:
#             item_dict (dict): [description]

#         Returns:
#             [type]: [description]
#         """
#         self._item_dict = item_dict
    
#     def __getattr__(self, k):
#         """[summary]

#         Args:
#             k ([type]): [description]

#         Raises:
#             AttributeError: [The value k is not in the study item dict]

#         Returns:
#             [torch.Tensor]: [The value for the study item]
#         """

#         val = self._item_dict.get(k)
#         if val is None:
#             raise AttributeError(f'{k} not in StudyItem - Keys={list(self._item_dict.keys())}')
        
#         return val
