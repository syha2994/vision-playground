from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import torch 
import gc
import os
from pathlib import Path
import random
import time

# GPU 상태 확인 및 출력
def print_gpu_status():
    """GPU 사용 상태 출력"""
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU 상태 확인")
        print("="*60)
        print(f"GPU 사용 가능: ✓")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 GPU: {torch.cuda.current_device()}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        
        # GPU 메모리 정보
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            free = total_memory - reserved
            
            print(f"\nGPU 메모리:")
            print(f"  전체 메모리: {total_memory:.2f} GB")
            print(f"  할당됨: {allocated:.2f} GB")
            print(f"  예약됨: {reserved:.2f} GB")
            print(f"  사용 가능: {free:.2f} GB")
            print(f"  사용률: {(reserved/total_memory)*100:.1f}%")
        print("="*60 + "\n")
    else:
        print("\n⚠ GPU를 사용할 수 없습니다! CPU 모드로 실행됩니다.\n")

def initialize_pipeline(use_cpu_offload=False):
    """
    파이프라인 초기화 함수 (__init__ 역할)
    
    Args:
        use_cpu_offload: True면 CPU offload 모드 (메모리 절약, 느림), False면 GPU 직접 로드 (빠름)
    
    Returns:
        초기화된 파이프라인 객체
    """
    # GPU 상태 출력
    print_gpu_status()
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    # 이미지 편집 파이프라인 로드
    print("파이프라인 로딩 중...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16
    )
    
    # GPU 메모리가 충분하면 CPU offload 없이 직접 GPU에 로드 (속도 향상)
    # CPU offload는 메모리 절약하지만 매번 GPU로 이동시켜야 해서 느림
    if use_cpu_offload:
        print("CPU offload 모드: 메모리 절약하지만 느릴 수 있습니다.")
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    else:
        print("GPU 직접 로드 모드: 빠르지만 메모리를 더 사용합니다.")
        pipe = pipe.to("cuda")
        # Attention slicing은 선택적으로 사용 (메모리 부족 시에만)
        # pipe.enable_attention_slicing()
    
    # Lightning LoRA 로드
    print("Lightning LoRA 로딩 중...")
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning", 
        weight_name="Qwen-Image-fp8-e4m3fn-Lightning-4steps-V1.0-fp32.safetensors",
        adapter_name="lightning"
    )
    
    # Multiple-angles LoRA 로드
    try:
        print("Multiple-angles LoRA 로딩 중...")
        pipe.load_lora_weights(
            "dx8152/Qwen-Edit-2509-Multiple-angles",
            weight_name="镜头转换.safetensors",
            adapter_name="multiple_angles",
            prefix=None
        )
        print("✓ Multiple-angles LoRA 로드 완료")
    except Exception as e:
        print(f"⚠ Multiple-angles LoRA 로드 실패: {e}")
        print("   Lightning LoRA만 사용합니다.")
    
    pipe.set_adapters(["lightning"], adapter_weights=[1.0])
    
    # 파이프라인 로드 완료 후 GPU 상태 확인
    print("\n파이프라인 로드 완료 후 GPU 상태:")
    print_gpu_status()
    
    return pipe

# 카메라 각도/위치 변경 프롬프트 목록 생성
def generate_camera_prompts(count=100):
    """다양한 카메라 각도/위치 변경 프롬프트 생성"""
    base_prompts = [
        # 기본 이동
        "Move the camera forward.",
        "Move the camera left.",
        "Move the camera right.",
        "Move the camera up.",
        "Move the camera down.",
        
        # 회전 (15도 단위)
        "Rotate the camera 15 degrees to the left.",
        "Rotate the camera 30 degrees to the left.",
        "Rotate the camera 45 degrees to the left.",
        "Rotate the camera 60 degrees to the left.",
        "Rotate the camera 75 degrees to the left.",
        "Rotate the camera 90 degrees to the left.",
        
        "Rotate the camera 15 degrees to the right.",
        "Rotate the camera 30 degrees to the right.",
        "Rotate the camera 45 degrees to the right.",
        "Rotate the camera 60 degrees to the right.",
        "Rotate the camera 75 degrees to the right.",
        "Rotate the camera 90 degrees to the right.",
        
        # 위아래 회전
        "Rotate the camera 15 degrees upward.",
        "Rotate the camera 30 degrees upward.",
        "Rotate the camera 45 degrees upward.",
        "Rotate the camera 60 degrees upward.",
        
        "Rotate the camera 15 degrees downward.",
        "Rotate the camera 30 degrees downward.",
        "Rotate the camera 45 degrees downward.",
        "Rotate the camera 60 degrees downward.",
        
        # 특정 뷰
        "Turn the camera to a top-down view.",
        "Turn the camera to a bottom-up view.",
        "Turn the camera to a wide-angle lens.",
        "Turn the camera to a close-up.",
        "Turn the camera to a side view.",
        "Turn the camera to a front view.",
        
        # 조합된 이동
        "Move the camera forward and rotate 30 degrees to the left.",
        "Move the camera forward and rotate 30 degrees to the right.",
        "Move the camera left and rotate 30 degrees upward.",
        "Move the camera right and rotate 30 degrees downward.",
        
        # 조금씩 다른 각도 (더 세밀한 변형)
        "Rotate the camera 10 degrees to the left.",
        "Rotate the camera 20 degrees to the left.",
        "Rotate the camera 25 degrees to the left.",
        "Rotate the camera 35 degrees to the left.",
        "Rotate the camera 50 degrees to the left.",
        
        "Rotate the camera 10 degrees to the right.",
        "Rotate the camera 20 degrees to the right.",
        "Rotate the camera 25 degrees to the right.",
        "Rotate the camera 35 degrees to the right.",
        "Rotate the camera 50 degrees to the right.",
        
        # 거리 조절
        "Move the camera closer.",
        "Move the camera farther away.",
        "Zoom in slightly.",
        "Zoom out slightly.",
        
        # 대각선 이동
        "Move the camera forward and to the left.",
        "Move the camera forward and to the right.",
        "Move the camera up and to the left.",
        "Move the camera up and to the right.",
        "Move the camera down and to the left.",
        "Move the camera down and to the right.",
    ]
    
    # 기본 프롬프트가 100개보다 적으면 반복 및 변형 추가
    prompts = []
    while len(prompts) < count:
        # 기본 프롬프트에서 랜덤 선택
        prompts.extend(base_prompts)
        
        # 추가 변형: 각도와 방향을 조금씩 변형
        for base in base_prompts[:20]:  # 처음 20개만 변형
            if "degrees" in base:
                # 각도를 약간 변경
                for offset in [-5, 5]:
                    new_prompt = base.replace("degrees", f"degrees (slightly adjusted)")
                    if new_prompt not in prompts:
                        prompts.append(new_prompt)
    
    # 정확히 count개만 선택 (랜덤 셔플)
    random.shuffle(prompts)
    return prompts[:count]


def process_images_in_folder(pipe, input_folder, output_folder, total_images_per_folder=120):
    """
    폴더 내 모든 하위 폴더와 이미지에 대해 각도 변형 생성
    입력 폴더 구조를 유지하면서 출력 폴더에도 동일한 구조로 생성
    각 폴더 내 이미지 개수에 따라 총 생성 개수를 균등 분배
    
    Args:
        pipe: 초기화된 파이프라인 객체
        input_folder: 입력 이미지가 있는 루트 폴더
        output_folder: 결과를 저장할 루트 폴더
        total_images_per_folder: 각 폴더당 생성할 총 이미지 수 (폴더 내 이미지 개수에 따라 분배)
    """
    # 입력 폴더 확인
    if not os.path.exists(input_folder):
        print(f"오류: 입력 폴더를 찾을 수 없습니다: {input_folder}")
        return
    
    # 출력 루트 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 입력 폴더를 Path 객체로 변환
    input_path = Path(input_folder)
    
    # 모든 하위 폴더와 이미지 파일 찾기 (재귀적으로)
    image_files_by_folder = {}
    
    # 하위 폴더들을 순회
    for subfolder in sorted(input_path.rglob('*')):
        if subfolder.is_dir():
            # 이 폴더 내의 이미지 파일 찾기
            image_files = []
            for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
                image_files.extend(subfolder.glob(f'*{ext}'))
            
            if image_files:
                # 입력 폴더 기준 상대 경로 계산
                relative_path = subfolder.relative_to(input_path)
                image_files_by_folder[relative_path] = image_files
    
    # 루트 폴더의 이미지도 확인
    root_image_files = []
    for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
        root_image_files.extend(input_path.glob(f'*{ext}'))
    
    if root_image_files:
        image_files_by_folder[Path('.')] = root_image_files
    
    # 전체 이미지 개수 계산
    total_images = sum(len(files) for files in image_files_by_folder.values())
    
    if total_images == 0:
        print(f"오류: {input_folder} 폴더에서 이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(image_files_by_folder)}개의 폴더에서 {total_images}개의 이미지 파일을 찾았습니다.")
    print(f"각 폴더당 총 {total_images_per_folder}장의 이미지를 생성합니다.")
    
    # 전체 이미지 인덱스 추적
    global_img_idx = 0
    
    # 각 폴더별로 처리
    for folder_idx, (relative_folder, image_files) in enumerate(image_files_by_folder.items(), 1):
        print(f"\n{'#'*60}")
        print(f"폴더 [{folder_idx}/{len(image_files_by_folder)}]: {relative_folder}")
        print(f"{'#'*60}")
        
        # 출력 폴더 경로 생성 (입력 구조 유지)
        if str(relative_folder) == '.':
            current_output_folder = output_folder
        else:
            current_output_folder = os.path.join(output_folder, str(relative_folder))
        
        # 출력 폴더 생성
        os.makedirs(current_output_folder, exist_ok=True)
        
        # 이 폴더 내 이미지 개수에 따라 총 120장을 분배
        num_images_in_folder = len(image_files)
        images_per_input = total_images_per_folder // num_images_in_folder
        remainder = total_images_per_folder % num_images_in_folder
        
        print(f"  폴더 내 이미지 개수: {num_images_in_folder}개")
        print(f"  각 이미지당 생성 개수: {images_per_input}장" + (f" (나머지 {remainder}장은 첫 번째 이미지들에 추가)" if remainder > 0 else ""))
        
        # 카메라 프롬프트 생성 (최대 필요 개수만큼)
        max_prompts_needed = images_per_input + (1 if remainder > 0 else 0)
        camera_prompts = generate_camera_prompts(max_prompts_needed)
        
        # 각 이미지 처리
        for img_idx, image_path in enumerate(image_files, 1):
            global_img_idx += 1
            print(f"\n{'='*60}")
            print(f"[전체: {global_img_idx}/{total_images}] [{img_idx}/{len(image_files)}] 처리 중: {image_path.name}")
            print(f"폴더: {relative_folder}")
            print(f"{'='*60}")
            
            try:
                # 이미지 로드
                input_image = Image.open(image_path).convert("RGB")
                
                # 원본 이미지 크기 가져오기
                original_width, original_height = input_image.size
                print(f"  원본 이미지 크기: {original_width}x{original_height}")
                
                # 이미지 리사이즈: 긴 변을 512로 맞추고 비율 유지
                max_dimension = 256
                if original_width >= original_height:
                    # 가로가 더 긴 경우
                    new_width = max_dimension
                    new_height = int(original_height * (max_dimension / original_width))
                else:
                    # 세로가 더 긴 경우
                    new_height = max_dimension
                    new_width = int(original_width * (max_dimension / original_height))
                
                # 이미지 리사이즈
                resized_image = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"  리사이즈된 크기: {new_width}x{new_height}")
                
                # 원본 이미지 이름 (확장자 제외)
                base_name = Path(image_path).stem
                
                # 이 이미지에 할당된 프롬프트 개수 계산
                # 나머지가 있으면 첫 번째 이미지들에 추가
                current_images_count = images_per_input
                if img_idx <= remainder:
                    current_images_count += 1
                
                # 이 이미지에 사용할 프롬프트 선택
                prompts_for_this_image = camera_prompts[:current_images_count]
                
                print(f"  이 이미지에 {current_images_count}장 생성 예정")
                
                # 각 프롬프트로 이미지 생성
                for prompt_idx, prompt in enumerate(prompts_for_this_image, 1):
                    try:
                        if prompt_idx % 10 == 0 or prompt_idx == 1:
                            print(f"  [{prompt_idx}/{current_images_count}] {prompt[:50]}...")
                        
                        # 이미지 생성 시간 측정
                        start_time = time.time()
                        
                        # 이미지 생성 (리사이즈된 크기로)
                        output_image = pipe(
                            image=[resized_image],
                            prompt=prompt,
                            num_inference_steps=8,
                            generator=torch.manual_seed(prompt_idx + global_img_idx * 1000)  # 재현 가능한 시드
                        ).images[0]
                        
                        # 생성 시간 계산
                        elapsed_time = time.time() - start_time
                        print(f"    생성 시간: {elapsed_time:.2f}초")
                        
                        # 파일명 생성
                        # 프롬프트를 파일명에 안전하게 사용하기 위해 간단하게 변형
                        prompt_safe = prompt.replace(" ", "_").replace(".", "").replace(",", "")[:50]
                        output_filename = f"{base_name}_var{prompt_idx:03d}_{prompt_safe}.png"
                        output_path = os.path.join(current_output_folder, output_filename)
                        
                        # 저장
                        output_image.save(output_path)
                        
                        # GPU 메모리 정리 (주기적으로)
                        if prompt_idx % 10 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                    except Exception as e:
                        print(f"    ✗ 프롬프트 {prompt_idx} 처리 실패: {e}")
                        continue
                
                print(f"✓ {image_path.name} 처리 완료: {current_images_count}장 생성")
                
            except Exception as e:
                print(f"✗ {image_path.name} 로드 실패: {e}")
                continue
    
    print(f"\n{'='*60}")
    print("모든 이미지 처리가 완료되었습니다!")
    print(f"출력 폴더: {output_folder}")
    print(f"입력 폴더 구조가 유지되었습니다.")
    print(f"{'='*60}")


def main():
    """메인 함수"""
    # 파이프라인 초기화 (__init__ 역할)
    use_cpu_offload = False  # True로 설정하면 메모리 절약하지만 느림
    pipe = initialize_pipeline(use_cpu_offload=use_cpu_offload)
    
    # 설정
    input_folder = "/home/crefle/data/digital_gauges/split_1"  # 입력 이미지 폴더
    output_folder = "/home/crefle/data/digital_gauges/split_1_variations"  # 출력 폴더
    total_images_per_folder = 120  # 각 폴더당 생성할 총 이미지 수 (폴더 내 이미지 개수에 따라 균등 분배)
    
    # 또는 직접 경로 입력
    # input_folder = input("입력 이미지 폴더 경로: ").strip()
    # output_folder = input("출력 폴더 경로: ").strip()
    
    # 이미지 처리 실행
    process_images_in_folder(pipe, input_folder, output_folder, total_images_per_folder)


if __name__ == "__main__":
    main()

