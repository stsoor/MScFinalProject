from copy import deepcopy

class CallableCoupling:
    def __init__(self, _callable, *args, _add_call_args_before=False, _add_call_args_after=False, _add_call_kwargs=False, **kwargs):
        self.callable = _callable
        self.add_call_args_before =_add_call_args_before
        self.add_call_args_after =_add_call_args_after
        self.add_call_kwargs =_add_call_kwargs
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, *args, **kwargs):
        all_args = self.args
        all_kwargs = deepcopy(self.kwargs)
        if self.add_call_args_before:
            all_args = args + all_args
        if self.add_call_args_after:
            all_args = all_args + args
        if self.add_call_kwargs:
            all_kwargs.update(kwargs)
        return self.callable(*all_args, **all_kwargs)
