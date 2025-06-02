from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

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

def load_predictor(name: str, prediction_length: int=1, target_dim: int=1, device_map: str="cuda"):
    """
    Load a model by its name.
    
    Args:
        name (str): The name of the model to load.
        
    Returns:
        object: The loaded model.
    """
    if  name in MOIRAI_MODELS:
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(name).to(device_map),
            prediction_length=prediction_length,
            context_length=4000,
            patch_size=32,
            num_samples=20,
            target_dim=target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    
    elif name in MOIRAI_MOE_MODELS:
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(name).to(device_map),
            prediction_length=prediction_length,
            context_length=4000,
            patch_size=32,
            num_samples=20,
            target_dim=target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    
    return model.create_predictor(batch_size=32)