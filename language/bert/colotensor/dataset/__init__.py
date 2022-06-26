
from language.bert.colotensor.dataset.wikitext import build_data_from_wikitext
from colossalai.core import global_context as gpc

_datasets = {
    "wikitext": build_data_from_wikitext,
}

def build_data(**args):
    if hasattr(gpc.config, "dataset"):
        assert (
            gpc.config.dataset in _datasets.keys()
        ), f"Invalid dataset name. dataset should be in {_datasets.keys()} or use default wikitext"
        builder = _datasets[gpc.config.dataset]
    else:
        builder = _datasets["wikitext"]
    return builder(**args)


__all__ = ["build_data"]