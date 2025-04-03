import os
import numpy as np
import cv2
import torch
import asyncio
from torchvision.transforms.functional import normalize
from concurrent.futures import ThreadPoolExecutor

from codeformer.basicsr.archs.rrdbnet_arch import RRDBNet
from codeformer.basicsr.utils import img2tensor, imwrite, tensor2img
from codeformer.basicsr.utils.download_util import load_file_from_url
from codeformer.basicsr.utils.realesrgan_utils import RealESRGANer
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.facelib.utils.misc import is_gray

# Create a thread pool executor for CPU-bound tasks
thread_pool = ThreadPoolExecutor()

pretrain_model_url = {
    "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "detection": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
    "realesrgan": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
}

async def download_weights():
    """Asynchronously download model weights if they don't exist"""
    download_tasks = []
    
    if not os.path.exists("CodeFormer/weights/CodeFormer/codeformer.pth"):
        download_tasks.append(
            asyncio.to_thread(
                load_file_from_url,
                url=pretrain_model_url["codeformer"],
                model_dir="CodeFormer/weights/CodeFormer",
                progress=True,
                file_name=None
            )
        )
    
    if not os.path.exists("CodeFormer/weights/facelib/detection_Resnet50_Final.pth"):
        download_tasks.append(
            asyncio.to_thread(
                load_file_from_url,
                url=pretrain_model_url["detection"],
                model_dir="CodeFormer/weights/facelib",
                progress=True,
                file_name=None
            )
        )
    
    if not os.path.exists("CodeFormer/weights/facelib/parsing_parsenet.pth"):
        download_tasks.append(
            asyncio.to_thread(
                load_file_from_url,
                url=pretrain_model_url["parsing"],
                model_dir="CodeFormer/weights/facelib",
                progress=True,
                file_name=None
            )
        )
    
    if not os.path.exists("CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth"):
        download_tasks.append(
            asyncio.to_thread(
                load_file_from_url,
                url=pretrain_model_url["realesrgan"],
                model_dir="CodeFormer/weights/realesrgan",
                progress=True,
                file_name=None
            )
        )
    
    if download_tasks:
        await asyncio.gather(*download_tasks)

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# set enhancer with RealESRGAN
def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler

# Initialize models - this can be done at module load time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These will be initialized in the init_models function
upsampler = None
codeformer_net = None

async def init_models():
    """Initialize models asynchronously"""
    global upsampler, codeformer_net
    
    # Download weights if needed
    await download_weights()
    
    # Set up models
    upsampler = await asyncio.to_thread(set_realesrgan)
    
    # Initialize CodeFormer model
    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    
    ckpt_path = "CodeFormer/weights/CodeFormer/codeformer.pth"
    
    # Load checkpoint in a separate thread to avoid blocking
    checkpoint = await asyncio.to_thread(lambda: torch.load(ckpt_path)["params_ema"])
    
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()
    
    os.makedirs("output", exist_ok=True)

async def process_face(face_helper, cropped_face, codeformer_fidelity):
    """Process a single face asynchronously"""
    # prepare data
    cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

    try:
        # Run inference in a non-blocking way
        def run_inference():
            with torch.no_grad():
                output = codeformer_net(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            return restored_face
        
        # Use asyncio.to_thread for Python 3.9+
        restored_face = await asyncio.to_thread(run_inference)
        
        torch.cuda.empty_cache()
    except RuntimeError as error:
        print(f"Failed inference for CodeFormer: {error}")
        restored_face = await asyncio.to_thread(
            lambda: tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
        )

    restored_face = restored_face.astype("uint8")
    return restored_face

async def inference_app(image, background_enhance, face_upsample, upscale, codeformer_fidelity):
    """Asynchronous version of the inference function"""
    global upsampler, codeformer_net
    
    # Initialize models if not already done
    if upsampler is None or codeformer_net is None:
        await init_models()
    
    # take the default setting for the demo
    has_aligned = False
    only_center_face = False
    draw_box = False
    detection_model = "retinaface_resnet50"
    
    img = image  # Directly use the input image array

    upscale = int(upscale)  # convert type to int
    if upscale > 4:  # avoid memory exceeded due to too large upscale
        upscale = 4
    if upscale > 2 and max(img.shape[:2]) > 1000:  # avoid memory exceeded due to too large img resolution
        upscale = 2
    if max(img.shape[:2]) > 1500:  # avoid memory exceeded due to too large img resolution
        upscale = 1
        background_enhance = False
        face_upsample = False

    # Create face helper in a non-blocking way
    def create_face_helper():
        return FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
    
    face_helper = await asyncio.to_thread(create_face_helper)
    bg_upsampler = upsampler if background_enhance else None
    face_upsampler = upsampler if face_upsample else None

    if has_aligned:
        # the input faces are already cropped and aligned
        def process_aligned():
            resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(resized_img, threshold=5)
            face_helper.cropped_faces = [resized_img]
            return resized_img
        
        img = await asyncio.to_thread(process_aligned)
    else:
        # Process image in a non-blocking way
        await asyncio.to_thread(face_helper.read_image, img)
        
        # get face landmarks for each face
        num_det_faces = await asyncio.to_thread(
            face_helper.get_face_landmarks_5,
            only_center_face=only_center_face,
            resize=640,
            eye_dist_threshold=5
        )
        
        # Sort faces by confidence if needed
        face_detections = face_helper.det_faces
        
        if len(face_detections) > 2:
            # Sort by confidence score only if more than 2 faces are detected
            sorted_faces = sorted(face_detections, key=lambda x: x[4], reverse=True)
            # Limit to the top two faces with the highest confidence scores
            sorted_faces = sorted_faces[:2]
            face_helper.det_faces = sorted_faces
        else:
            face_helper.det_faces = face_detections
        
        # align and warp each face
        await asyncio.to_thread(face_helper.align_warp_face)

    # Process faces in parallel
    face_tasks = []
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        face_tasks.append(process_face(face_helper, cropped_face, codeformer_fidelity))
    
    # Wait for all faces to be processed
    restored_faces = await asyncio.gather(*face_tasks)
    
    # Add restored faces to face helper
    for restored_face in restored_faces:
        face_helper.add_restored_face(restored_face)

    # paste_back
    if not has_aligned:
        # upsample the background
        if bg_upsampler is not None:
            # Use a thread for background upsampling
            def enhance_background():
                return bg_upsampler.enhance(img, outscale=upscale)[0]
            
            bg_img = await asyncio.to_thread(enhance_background)
        else:
            bg_img = None
        
        await asyncio.to_thread(face_helper.get_inverse_affine, None)
        
        # paste each restored face to the input image
        if face_upsample and face_upsampler is not None:
            def paste_faces_with_upsampler():
                return face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            
            restored_img = await asyncio.to_thread(paste_faces_with_upsampler)
        else:
            def paste_faces():
                return face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, 
                    draw_box=draw_box
                )
            
            restored_img = await asyncio.to_thread(paste_faces)
    
    return restored_img
