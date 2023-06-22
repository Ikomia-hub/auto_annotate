import os
import urllib
import requests
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

def load_grounding_dino(name, device):
    url_base = "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    if name == "Swin-T":
        model_file_name = "groundingdino_swint_ogc.pth"
        config_file_name = "GroundingDINO_SwinT_OGC.py"
        url_ext = "v0.1.0-alpha/groundingdino_swint_ogc.pth"

    if name == "Swin-B":
        model_file_name = "groundingdino_swinb_cogcoor.pth"
        config_file_name = "GroundingDINO_SwinB_cfg.py"
        url_ext = "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    

    parent_directory = os.path.join(os.path.dirname(os.path.dirname(
                            os.path.realpath(__file__))))

    model_config = os.path.join(parent_directory,
                                "GroundingDINO",
                                "groundingdino",
                                "config",
                                config_file_name
                            )

    model_weigth = os.path.join(
                                parent_directory,
                                "weights",
                                model_file_name,
                            )

    # Download model weight if not exist
    weights_folder = os.path.join(parent_directory, "weights")
    if not os.path.isdir(weights_folder):
        os.mkdir(weights_folder)

    if not os.path.isfile(model_weigth):
        url = os.path.join(url_base, url_ext)
        print("Downloading Grounding Dino model weight from {}".format(url))
        file_path = os.path.join(weights_folder, model_file_name)
        urllib.request.urlretrieve(url, file_path)
        print("Download completed!")

    model_dino = Model(
                model_config_path=model_config,
                model_checkpoint_path=model_weigth,
                device=device
    )

    return model_dino

def load_sam_predictor(name, device):
    base_url= "https://dl.fbaipublicfiles.com/segment_anything/"
    model_list = {"vit_b": "sam_vit_b_01ec64.pth",
                    "vit_l": "sam_vit_l_0b3195.pth",
                    "vit_h": "sam_vit_h_4b8939.pth"}
    
    parent_directory = os.path.join(os.path.dirname(os.path.dirname(
                            os.path.realpath(__file__))))
    model_folder = os.path.join(parent_directory, "weights")

    model_weight =  os.path.join(str(model_folder), model_list[name])

    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    if not os.path.isfile(model_weight):
        print("Downloading the SAM model...")
        model_url = base_url + model_list[name]
        response = requests.get(model_url)
        with open(os.path.join(model_folder, model_list[name]) , 'wb') as f:
            f.write(response.content)

    sam = sam_model_registry[name](checkpoint=model_weight)
    sam_pred = SamPredictor(sam)
    
    return sam_pred 