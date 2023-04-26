from fvcore.common.config import CfgNode as _CfgNode

class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    2. Support config versioning.
       When attempting to merge an old config, it will convert the old config automatically.

    .. automethod:: clone
    .. automethod:: freeze
    .. automethod:: defrost
    .. automethod:: is_frozen
    .. automethod:: load_yaml_with_base
    .. automethod:: merge_from_list
    .. automethod:: merge_from_other_cfg
    """

    def merge_from_dict(self, dict):
        pass
    
node = CfgNode()