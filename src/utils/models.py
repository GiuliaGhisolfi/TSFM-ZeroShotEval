from uni2ts.src.moirai.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.src.moirai.model.moirai_moe import (MoiraiMoEForecast,
                                                MoiraiMoEModule)

MOIRAI_MODELS = [
    "Salesforce/moirai-1.0-R-small",
    "Salesforce/moirai-1.0-R-base",
    "Salesforce/moirai-1.0-R-large",

    "Salesforce/moirai-1.1-R-small",
    "Salesforce/moirai-1.1-R-base",
    "Salesforce/moirai-1.1-R-large"
]
MOIRAI_MOE_MODELS = [
    "Salesforce/moirai-moe-1.0-R-small",
    "Salesforce/moirai-moe-1.0-R-base"
]

def load_model(name: str, prediction_length: int = 1):
    """
    Load a model by its name.
    
    Args:
        name (str): The name of the model to load.
        
    Returns:
        object: The loaded model.
    """
    if  name in MOIRAI_MODELS:
        return MoiraiForecast(
            module=MoiraiModule.from_pretrained(name),
            prediction_length=prediction_length,
            context_length=4000,
            patch_size=32,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    
    elif name in MOIRAI_MOE_MODELS:
        return MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(name),
            prediction_length=prediction_length,
            context_length=4000,
            patch_size=32,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )